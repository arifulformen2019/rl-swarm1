import logging
import os
import sys
import time
import gc
import threading
import ctypes
import weakref
import signal
from collections import defaultdict
from pathlib import Path

from genrl.blockchain import SwarmCoordinator
from genrl.communication import Communication
from genrl.communication.hivemind.hivemind_backend import HivemindBackend
from genrl.data import DataManager
from genrl.game import BaseGameManager
from genrl.game.game_manager import DefaultGameManagerMixin
from genrl.logging_utils.global_defs import get_logger
from genrl.logging_utils.system_utils import get_system_info
from genrl.rewards import RewardManager
from genrl.roles import RoleManager
from genrl.state import GameState
from genrl.trainer import TrainerModule
from huggingface_hub import login, whoami

from rgym_exp.src.utils.name_utils import get_name_from_peer_id

# Torch import for cleanup
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Colorful logging
try:
    from colorama import Fore, Style, init
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False
    class MockFore:
        CYAN = GREEN = RED = YELLOW = MAGENTA = BLUE = ""
    class MockStyle:
        RESET_ALL = ""
    Fore = MockFore()
    Style = MockStyle()

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class SystemMemoryMonitor:
    """Advanced system memory monitoring with emergency shutdown"""
    
    def __init__(self, manager_name="SwarmManager"):
        self.manager_name = manager_name
        self.baseline_memory = self._get_memory_info()
        self.peak_system_usage = self.baseline_memory['system_used_pct']
        self.memory_samples = []
        self.emergency_shutdown_threshold = 95  # 95% system RAM = emergency shutdown
        self.critical_threshold = 90  # 90% system RAM = critical
        self.warning_threshold = 85   # 85% system RAM = warning
        
        # Emergency state
        self.emergency_mode = False
        self.critical_hits = 0
        self.last_emergency_cleanup = 0
        
        # Monitoring thread
        self.monitoring_active = True
        self.monitor_thread = None
        self._start_monitoring()
    
    def _get_memory_info(self):
        """Get comprehensive memory information"""
        if not PSUTIL_AVAILABLE:
            return {'system_used_pct': 0, 'process_ram_gb': 0, 'system_available_gb': 0}
        
        try:
            process = psutil.Process()
            system_memory = psutil.virtual_memory()
            
            return {
                'process_ram_gb': process.memory_info().rss / 1024**3,
                'system_used_pct': system_memory.percent,
                'system_total_gb': system_memory.total / 1024**3,
                'system_available_gb': system_memory.available / 1024**3,
                'system_used_gb': system_memory.used / 1024**3,
            }
        except Exception as e:
            get_logger().debug(f"Memory info failed: {e}")
            return {'system_used_pct': 0, 'process_ram_gb': 0, 'system_available_gb': 0}
    
    def _start_monitoring(self):
        """Start background memory monitoring"""
        def monitor_loop():
            while self.monitoring_active:
                try:
                    self._check_system_memory()
                    time.sleep(30)  # Check every 30 seconds
                except Exception as e:
                    get_logger().error(f"Memory monitor error: {e}")
                    time.sleep(60)  # Back off on error
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        get_logger().info(f"{Fore.GREEN}🔍 [SYSTEM MONITOR] Background memory monitoring started{Style.RESET_ALL}")
    
    def _check_system_memory(self):
        """Check system memory and take action if needed"""
        memory_info = self._get_memory_info()
        system_used_pct = memory_info['system_used_pct']
        process_ram_gb = memory_info['process_ram_gb']
        available_gb = memory_info['system_available_gb']
        
        # Update peak
        if system_used_pct > self.peak_system_usage:
            self.peak_system_usage = system_used_pct
        
        # Add to samples
        self.memory_samples.append(system_used_pct)
        if len(self.memory_samples) > 100:
            self.memory_samples = self.memory_samples[-50:]
        
        # EMERGENCY SHUTDOWN CHECK
        if system_used_pct >= self.emergency_shutdown_threshold:
            self._emergency_shutdown(system_used_pct, available_gb)
            return
        
        # CRITICAL MEMORY CHECK
        elif system_used_pct >= self.critical_threshold:
            self.critical_hits += 1
            get_logger().error(
                f"{Fore.RED}🚨 CRITICAL SYSTEM MEMORY: {system_used_pct:.1f}% "
                f"({available_gb:.1f}GB free) - Hit #{self.critical_hits} - NUCLEAR CLEANUP{Style.RESET_ALL}"
            )
            self._nuclear_system_cleanup()
            
        # WARNING CHECK
        elif system_used_pct >= self.warning_threshold:
            get_logger().warning(
                f"{Fore.YELLOW}⚠️ HIGH SYSTEM MEMORY: {system_used_pct:.1f}% "
                f"({available_gb:.1f}GB free) - Process: {process_ram_gb:.1f}GB{Style.RESET_ALL}"
            )
            
        # NORMAL LOGGING (less frequent)
        else:
            # Only log every 10 minutes when normal
            if not hasattr(self, '_last_normal_log'):
                self._last_normal_log = 0
            
            if time.time() - self._last_normal_log > 600:  # 10 minutes
                get_logger().info(
                    f"{Fore.GREEN}📊 [SYSTEM MEMORY] {system_used_pct:.1f}% used | "
                    f"Process: {process_ram_gb:.1f}GB | Available: {available_gb:.1f}GB{Style.RESET_ALL}"
                )
                self._last_normal_log = time.time()
    
    def _emergency_shutdown(self, system_used_pct, available_gb):
        """Emergency shutdown when system memory critical"""
        get_logger().error(
            f"{Fore.RED}💥 EMERGENCY SHUTDOWN: System RAM at {system_used_pct:.1f}% "
            f"(only {available_gb:.1f}GB free) - SHUTTING DOWN TO PREVENT SYSTEM CRASH{Style.RESET_ALL}"
        )
        
        try:
            # Try to save state before shutdown
            self._emergency_save_state()
            
            # Nuclear cleanup attempt
            self._nuclear_system_cleanup()
            
        except Exception as e:
            get_logger().error(f"Emergency cleanup failed: {e}")
        
        finally:
            # Force shutdown
            get_logger().error(f"{Fore.RED}💥 FORCED SHUTDOWN - System memory exhausted{Style.RESET_ALL}")
            os._exit(1)  # Hard exit
    
    def _emergency_save_state(self):
        """Try to save critical state before emergency shutdown"""
        try:
            get_logger().info(f"{Fore.YELLOW}💾 [EMERGENCY SAVE] Attempting to save state{Style.RESET_ALL}")
            # This would be implemented based on what state needs saving
            pass
        except Exception as e:
            get_logger().error(f"Emergency save failed: {e}")
    
    def _nuclear_system_cleanup(self):
        """Nuclear system-wide memory cleanup"""
        current_time = time.time()
        
        # Prevent too frequent nuclear cleanups
        if current_time - self.last_emergency_cleanup < 60:  # 1 minute cooldown
            return
        
        self.last_emergency_cleanup = current_time
        
        try:
            get_logger().warning(f"{Fore.RED}💥 [NUCLEAR SYSTEM] Nuclear system cleanup initiated{Style.RESET_ALL}")
            
            # 1. Python memory arenas
            if hasattr(sys, 'intern'):
                sys.intern.__dict__.clear()
            
            # 2. Module cache nuclear cleanup
            modules_to_clear = []
            for name, module in list(sys.modules.items()):
                if any(pattern in name.lower() for pattern in [
                    'transformers', 'datasets', 'torch', 'numpy',
                    '_cache', 'cache', 'temp', 'huggingface'
                ]):
                    if hasattr(module, '__dict__'):
                        try:
                            module.__dict__.clear()
                            modules_to_clear.append(name)
                        except:
                            pass
            
            # Remove cleared modules
            for name in modules_to_clear:
                try:
                    del sys.modules[name]
                except:
                    pass
            
            # 3. Nuclear garbage collection
            total_collected = 0
            for generation in [2, 1, 0]:
                for _ in range(10):  # Reduced from 50 to 10
                    collected = gc.collect(generation)
                    total_collected += collected
                    if collected == 0:
                        break
            
            # 4. OS-level memory operations
            if sys.platform.startswith('linux'):
                try:
                    # Sync filesystems
                    os.system('sync')
                    
                    # Massive malloc_trim
                    libc = ctypes.CDLL("libc.so.6")
                    for _ in range(50):  # Reduced from 200 to 50
                        libc.malloc_trim(0)
                        if _ % 10 == 9:
                            time.sleep(0.01)  # Brief pause every 10 calls
                    
                    # Try to drop caches (requires root, but try anyway)
                    try:
                        with open('/proc/sys/vm/drop_caches', 'w') as f:
                            f.write('3')  # Drop all caches
                    except (PermissionError, FileNotFoundError):
                        pass
                    
                    # Memory compaction
                    try:
                        with open('/proc/sys/vm/compact_memory', 'w') as f:
                            f.write('1')
                    except (PermissionError, FileNotFoundError):
                        pass
                        
                except Exception as e:
                    get_logger().debug(f"OS memory operations failed: {e}")
            
            get_logger().info(
                f"{Fore.GREEN}✅ [NUCLEAR SYSTEM] Nuclear cleanup completed - "
                f"{total_collected} objects collected{Style.RESET_ALL}"
            )
            
        except Exception as e:
            get_logger().error(f"Nuclear system cleanup failed: {e}")
    
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)


class SwarmGameManager(BaseGameManager, DefaultGameManagerMixin):
    """Enhanced GameManager with nuclear memory protection"""

    def __init__(
        self,
        coordinator: SwarmCoordinator,
        max_stage: int,
        max_round: int,
        game_state: GameState,
        reward_manager: RewardManager,
        trainer: TrainerModule,
        data_manager: DataManager,
        communication: Communication,
        role_manager: RoleManager | None = None,
        run_mode: str = "train",
        log_dir: str = "logs",
        hf_token: str | None = None,
        hf_push_frequency: int = 20,
        **kwargs,
    ):

        super().__init__(
            max_stage=max_stage,
            max_round=max_round,
            game_state=game_state,
            reward_manager=reward_manager,
            trainer=trainer,
            data_manager=data_manager,
            communication=communication,
            role_manager=role_manager,
            run_mode=run_mode,
        )

        # Enhanced memory management
        self.system_monitor = SystemMemoryMonitor("SwarmGameManager")
        self.round_counter = 0
        self.agent_block_counter = 0
        self.last_memory_cleanup = time.time()
        self.last_rewards_cleanup = time.time()
        
        # Memory thresholds - increased to be less aggressive
        self.process_memory_threshold = 12.0  # 12GB process memory
        self.nuclear_process_threshold = 18.0  # 18GB nuclear cleanup
        
        # Training state
        self.training_active = True
        self.emergency_mode = False
        
        # Component setup
        assert isinstance(self.communication, HivemindBackend)
        self.train_timeout = 60 * 60 * 24 * 31  # 1 month

        # Peer and model setup
        self.peer_id = self.communication.get_id()
        self.state.peer_id = self.peer_id
        self.animal_name = get_name_from_peer_id(self.peer_id, True)
        
        # Model name for logging
        model_name = self._get_model_name()
        self.model_display_name = self._clean_model_name(model_name)
        
        # Logging setup
        self._setup_logging(log_dir, model_name)
        
        # Coordinator setup
        self.coordinator = coordinator
        self.coordinator.register_peer(self.peer_id)
        round, _ = self.coordinator.get_round_and_stage()
        self.state.round = round
        self.communication.step_ = self.state.round

        # HuggingFace setup
        self.hf_token = hf_token
        self.hf_push_frequency = hf_push_frequency
        self._setup_huggingface(model_name)

        # Blockchain submission - Fixed frequency
        self.batched_signals = 0.0
        self.time_since_submit = time.time()
        self.submit_period = 1.0  # 6 minutes instead of 1 hour
        self.submitted_this_round = False

        # Setup emergency handlers
        self._setup_emergency_handlers()

        get_logger().info(
            f"🐱 Hello 🐈 [{self.animal_name}] 🦮 [{self.peer_id}]!"
        )
        get_logger().info(f"bootnodes: {kwargs.get('bootnodes', [])}")

        with open(os.path.join(log_dir, f"system_info.txt"), "w") as f:
            f.write(get_system_info())

        get_logger().info(
            f"{Fore.GREEN}🛡️ [NUCLEAR MANAGER] Nuclear memory protection enabled:\n"
            f"   🔍 System monitoring: Active\n"
            f"   🚨 Emergency at: 90% system RAM\n"
            f"   💥 Shutdown at: 95% system RAM\n"
            f"   🧹 Process threshold: {self.process_memory_threshold}GB\n"
            f"   💣 Nuclear threshold: {self.nuclear_process_threshold}GB{Style.RESET_ALL}"
        )

    def _get_model_name(self):
        """Safely get model name"""
        try:
            if hasattr(self.trainer, "use_vllm") and self.trainer.use_vllm:
                return getattr(self.trainer, "model_name", "vLLM_Model")
            else:
                config_obj = getattr(getattr(self.trainer, "model", None), "config", None)
                if config_obj:
                    return getattr(config_obj, "_name_or_path", "UnknownModel")
                return "UnknownModel"
        except Exception:
            return "UnknownModel"

    def _clean_model_name(self, model_name):
        """Clean model name for display"""
        if "/" in model_name:
            clean_name = model_name.split("/")[-1]
        else:
            clean_name = model_name
            
        # Remove common suffixes
        clean_suffixes = ["-Instruct", "-Chat", "-Base", "-v1", "-v2", "-v3"]
        for suffix in clean_suffixes:
            if clean_name.endswith(suffix):
                clean_name = clean_name[:-len(suffix)]
                break
        
        return clean_name

    def _setup_logging(self, log_dir, model_name):
        """Setup logging with model name prefix"""
        format_msg = f"[{self.model_display_name}] %(asctime)s %(levelname)s: %(message)s"
        logging.basicConfig(level=logging.INFO, format=format_msg)
        formatter = logging.Formatter(format_msg)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f"training_{self.animal_name}.log")
        )
        file_handler.setFormatter(formatter)
        get_logger().addHandler(file_handler)
        
        get_logger().info(f"Using Model: {model_name}")

    def _setup_huggingface(self, model_name):
        """Setup HuggingFace integration"""
        if self.hf_token not in [None, "None"]:
            if not (hasattr(self.trainer, "use_vllm") and self.trainer.use_vllm):
                try:
                    username = whoami(token=self.hf_token)["name"]
                    model_name_suffix = model_name.split("/")[-1]
                    hub_model_id = f"{username}/{model_name_suffix}-Gensyn-Swarm-{self.animal_name}"
                    
                    self.trainer.args.hub_model_id = hub_model_id
                    self.trainer.args.push_to_hub = True
                    self.trainer.args.hub_token = self.hf_token
                    
                    get_logger().info("Logging into Hugging Face Hub...")
                    login(self.hf_token)
                except Exception as e:
                    get_logger().warning(f"Could not set up Hugging Face push. Error: {e}")
            else:
                get_logger().info("Hugging Face push is disabled in vLLM mode.")

    def _setup_emergency_handlers(self):
        """Setup emergency signal handlers"""
        def emergency_handler(signum, frame):
            get_logger().error(f"{Fore.RED}🚨 [EMERGENCY] Received signal {signum} - Emergency cleanup{Style.RESET_ALL}")
            self._emergency_cleanup_and_exit()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, emergency_handler)
        signal.signal(signal.SIGINT, emergency_handler)

    def _emergency_cleanup_and_exit(self):
        """Emergency cleanup and exit"""
        try:
            get_logger().error(f"{Fore.RED}💥 [EMERGENCY EXIT] Starting emergency cleanup{Style.RESET_ALL}")
            
            # Stop training
            self.training_active = False
            
            # Stop memory monitoring
            self.system_monitor.stop_monitoring()
            
            # Try to save state
            self._save_to_hf()
            
            # Nuclear cleanup
            self._nuclear_manager_cleanup()
            
        except Exception as e:
            get_logger().error(f"Emergency cleanup failed: {e}")
        finally:
            get_logger().error(f"{Fore.RED}💥 [EMERGENCY EXIT] Exiting{Style.RESET_ALL}")
            os._exit(1)

    def _get_process_memory_gb(self):
        """Get current process memory in GB"""
        if PSUTIL_AVAILABLE:
            try:
                return psutil.Process().memory_info().rss / 1024**3
            except:
                return 0.0
        return 0.0

    def _check_process_memory_emergency(self, context=""):
        """Check process memory and handle emergencies"""
        memory_gb = self._get_process_memory_gb()
        
        if memory_gb > self.nuclear_process_threshold:
            get_logger().error(
                f"{Fore.RED}💥 [PROCESS NUCLEAR] {memory_gb:.1f}GB - NUCLEAR CLEANUP{Style.RESET_ALL}"
            )
            self._nuclear_manager_cleanup()
            self.emergency_mode = True
            return True
            
        elif memory_gb > self.process_memory_threshold:
            get_logger().warning(
                f"{Fore.YELLOW}⚠️ [PROCESS HIGH] {memory_gb:.1f}GB - Enhanced cleanup{Style.RESET_ALL}"
            )
            self._enhanced_manager_cleanup()
            return False
            
        return False

    def _nuclear_manager_cleanup(self):
        """Nuclear manager-level cleanup - Less aggressive"""
        try:
            get_logger().error(f"{Fore.RED}💥 [NUCLEAR MANAGER] Nuclear cleanup initiated{Style.RESET_ALL}")
            
            initial_memory = self._get_process_memory_gb()
            
            # 1. Clear only old rewards history (keep recent)
            self._safe_rewards_cleanup()
            
            # 2. Clear communication buffers
            self._safe_communication_cleanup()
            
            # 3. Clear trainer caches
            if hasattr(self.trainer, 'cleanup'):
                self.trainer.cleanup()
            
            # 4. Clear data manager caches
            if hasattr(self.data_manager, 'cleanup'):
                self.data_manager.cleanup()
            
            # 5. Clear only old game state trees
            self._safe_gamestate_cleanup()
            
            # 6. System-level cleanup
            self.system_monitor._nuclear_system_cleanup()
            
            final_memory = self._get_process_memory_gb()
            freed = initial_memory - final_memory
            
            get_logger().info(
                f"{Fore.GREEN}✅ [NUCLEAR MANAGER] Nuclear cleanup completed: {freed:.1f}GB freed{Style.RESET_ALL}"
            )
            
        except Exception as e:
            get_logger().error(f"Nuclear manager cleanup failed: {e}")

    def _enhanced_manager_cleanup(self):
        """Enhanced manager cleanup for high memory situations"""
        try:
            get_logger().info(f"{Fore.CYAN}🧹 [ENHANCED CLEANUP] Enhanced manager cleanup{Style.RESET_ALL}")
            
            # 1. Safe rewards cleanup
            self._safe_rewards_cleanup()
            
            # 2. Communication cleanup
            self._safe_communication_cleanup()
            
            # 3. Game state cleanup
            self._safe_gamestate_cleanup()
            
            # 4. Python memory cleanup
            self._python_memory_cleanup()
            
        except Exception as e:
            get_logger().error(f"Enhanced manager cleanup failed: {e}")

    def _safe_rewards_cleanup(self):
        """Safe rewards cleanup that preserves recent data"""
        try:
            if hasattr(self, 'rewards') and len(self.rewards) > 20:
                # Keep only recent 20 rounds instead of 10
                recent_keys = list(self.rewards.keys())[-20:]
                old_rewards = dict(self.rewards)
                self.rewards.clear()
                
                for key in recent_keys:
                    if key in old_rewards:
                        self.rewards[key] = old_rewards[key]
                
                get_logger().info(f"{Fore.CYAN}🧹 [SAFE] Rewards trimmed to recent 20 rounds{Style.RESET_ALL}")
                
        except Exception as e:
            get_logger().debug(f"Safe rewards cleanup failed: {e}")

    def _safe_communication_cleanup(self):
        """Safe communication cleanup"""
        try:
            # Only clear non-essential buffers
            if hasattr(self.communication, '_message_cache'):
                self.communication._message_cache.clear()
            
            # Clear old peer information
            if hasattr(self.communication, '_old_peers'):
                self.communication._old_peers.clear()
                
        except Exception as e:
            get_logger().debug(f"Safe communication cleanup failed: {e}")

    def _safe_gamestate_cleanup(self):
        """Safe game state cleanup"""
        try:
            # Clear only old game trees, keep recent ones
            if hasattr(self.state, 'trees'):
                # Keep trees for current round only
                current_round = self.state.round
                trees_to_remove = []
                
                for agent in list(self.state.trees.keys()):
                    agent_trees = self.state.trees[agent]
                    if isinstance(agent_trees, dict):
                        # Clear old batches, keep recent
                        if len(agent_trees) > 10:  # Keep only 10 most recent instead of 5
                            batch_keys = list(agent_trees.keys())
                            for old_batch in batch_keys[:-10]:
                                del agent_trees[old_batch]
                
        except Exception as e:
            get_logger().debug(f"Safe gamestate cleanup failed: {e}")

    def _python_memory_cleanup(self):
        """Python-level memory cleanup"""
        try:
            # Garbage collection
            total_collected = 0
            for generation in [2, 1, 0]:
                for _ in range(3):  # Reduced from 5 to 3
                    collected = gc.collect(generation)
                    total_collected += collected
                    if collected == 0:
                        break
            
            # OS memory return
            if sys.platform.startswith('linux'):
                try:
                    libc = ctypes.CDLL("libc.so.6")
                    for _ in range(10):  # Reduced from 20 to 10
                        libc.malloc_trim(0)
                except:
                    pass
            
            if total_collected > 0:
                get_logger().info(f"{Fore.GREEN}🧹 [PYTHON] {total_collected} objects collected{Style.RESET_ALL}")
                
        except Exception as e:
            get_logger().debug(f"Python cleanup failed: {e}")

    def _get_total_rewards_by_agent(self):
        """Get total rewards by agent with memory safety"""
        rewards_by_agent = defaultdict(int)
        
        try:
            for stage in range(self.state.stage):
                if stage in self.rewards:
                    rewards = self.rewards[stage]
                    for agent_id, agent_rewards in rewards.items():
                        for batch_id, batch_rewards in agent_rewards.items():
                            tot = 0
                            for generation_rewards in batch_rewards:
                                tot += sum(generation_rewards)
                            rewards_by_agent[agent_id] += tot
        except Exception as e:
            get_logger().debug(f"Error calculating rewards: {e}")
        
        return rewards_by_agent

    def _get_my_rewards(self, signal_by_agent):
        """Get rewards for this agent - Fixed logic"""
        if len(signal_by_agent) == 0:
            return 0
        
        if self.peer_id in signal_by_agent:
            return signal_by_agent[self.peer_id]  # Return actual signal value
        else:
            return 0

    def _try_submit_to_chain(self, signal_by_agent):
        """Try to submit rewards to blockchain with enhanced logging"""
        elapsed_time_hours = (time.time() - self.time_since_submit) / 3600
        
        if elapsed_time_hours > self.submit_period:
            try:
                get_logger().info(
                    f"{Fore.CYAN}🚀 [SUBMIT STARTING] Round: {self.state.round} | "
                    f"Points: {int(self.batched_signals)} | Agent: {self.animal_name}{Style.RESET_ALL}"
                )
                
                # Submit reward
                self.coordinator.submit_reward(
                    self.state.round, 0, int(self.batched_signals), self.peer_id
                )
                
                # Determine winner
                if len(signal_by_agent) > 0:
                    max_agent, max_signal = max(signal_by_agent.items(), key=lambda x: x[1])
                    try:
                        winner_name = get_name_from_peer_id(max_agent, True) if max_agent != self.peer_id else self.animal_name
                    except:
                        winner_name = "unknown-agent"
                else:
                    max_agent = self.peer_id
                    winner_name = self.animal_name
                    max_signal = int(self.batched_signals)

                # Submit winners
                self.coordinator.submit_winners(self.state.round, [max_agent], self.peer_id)
                
                get_logger().info(
                    f"{Fore.GREEN}✅ [SUBMIT SUCCESS] 🎉 POINTS SUBMITTED! 🎉\n"
                    f"   💰 Points Sent: {int(self.batched_signals)}\n"
                    f"   🏆 Round Winner: {winner_name} ({max_signal} pts)\n"
                    f"   🕐 Next Submit: {self.submit_period} hours\n"
                    f"   🐾 Agent: {self.animal_name}{Style.RESET_ALL}"
                )
                
                # Reset counters
                submitted_points = int(self.batched_signals)
                self.batched_signals = 0.0
                self.time_since_submit = time.time()
                self.submitted_this_round = True
                
                get_logger().info(
                    f"{Fore.BLUE}📊 [STATS] Total Submitted: {submitted_points} | "
                    f"Round: {self.state.round}{Style.RESET_ALL}"
                )
                
            except Exception as e:
                get_logger().error(
                    f"{Fore.RED}❌ [SUBMIT FAILED] 💥 SUBMISSION ERROR! 💥\n"
                    f"   🚨 Error: {str(e)}\n"
                    f"   💰 Points Lost: {int(self.batched_signals)}\n"
                    f"   🐾 Agent: {self.animal_name}{Style.RESET_ALL}"
                )
                
        else:
            remaining_hours = self.submit_period - elapsed_time_hours
            remaining_minutes = remaining_hours * 60
            
            # Only log every 30 minutes when waiting
            if not hasattr(self, '_last_waiting_log'):
                self._last_waiting_log = 0
            
            if time.time() - self._last_waiting_log > 1800:  # 30 minutes
                get_logger().info(
                    f"{Fore.YELLOW}⏳ [WAITING] Next submit in: {remaining_minutes:.0f} minutes | "
                    f"Current points: {int(self.batched_signals)} | Agent: {self.animal_name}{Style.RESET_ALL}"
                )
                self._last_waiting_log = time.time()

    def _hook_after_rewards_updated(self):
        """Hook after rewards updated with memory management"""
        try:
            signal_by_agent = self._get_total_rewards_by_agent()
            old_signals = self.batched_signals
            self.batched_signals += self._get_my_rewards(signal_by_agent)
            
            # Log reward updates
            reward_gained = self.batched_signals - old_signals
            if reward_gained > 0:
                get_logger().info(
                    f"{Fore.GREEN}💰 [REWARD GAINED] +{reward_gained:.1f} points | "
                    f"Total: {int(self.batched_signals)} | Agent: {self.animal_name}{Style.RESET_ALL}"
                )
            
            self._try_submit_to_chain(signal_by_agent)
            
            # Memory management after rewards - Less frequent
            current_time = time.time()
            if current_time - self.last_rewards_cleanup > 600:  # Every 10 minutes instead of 5
                self._check_process_memory_emergency(" - REWARDS")
                self.last_rewards_cleanup = current_time
                
        except Exception as e:
            get_logger().error(f"Rewards hook failed: {e}")

    def _hook_after_round_advanced(self):
        """Enhanced round advancement with nuclear memory protection"""
        self.round_counter += 1
        
        get_logger().info(
            f"{Fore.MAGENTA}🔄 [ROUND ADVANCED] 🚀 NEW ROUND STARTED! 🚀\n"
            f"   📈 Round: {self.state.round}\n"  
            f"   🎯 Total Rounds: {self.round_counter}\n"
            f"   💰 Pending Points: {int(self.batched_signals)}\n"
            f"   🐾 Agent: {self.animal_name}{Style.RESET_ALL}"
        )
        
        self._save_to_hf()

        if not self.submitted_this_round:
            signal_by_agent = self._get_total_rewards_by_agent()
            self._try_submit_to_chain(signal_by_agent)
        
        self.submitted_this_round = False

        # Memory management schedule - Less aggressive
        try:
            # Check process memory
            memory_emergency = self._check_process_memory_emergency(" - ROUND END")
            
            # Light cleanup every round
            gc.collect()
            
            # Enhanced cleanup schedule based on memory pressure
            if memory_emergency or self.round_counter % 10 == 0:  # Every 10 rounds instead of 5
                self._enhanced_manager_cleanup()
                
            # Nuclear cleanup for extreme cases - Much less frequent
            if self.round_counter % 100 == 0:  # Every 100 rounds instead of 25
                get_logger().info(f"{Fore.RED}💥 [NUCLEAR SCHEDULE] Round {self.round_counter} - Nuclear cleanup{Style.RESET_ALL}")
                self._nuclear_manager_cleanup()
                
        except Exception as e:
            get_logger().error(f"Round cleanup failed: {e}")

        # Block until swarm round advances
        self.agent_block()

    def _hook_after_game(self):
        """Hook after game ends"""
        try:
            self._save_to_hf()
            
            get_logger().info(
                f"{Fore.GREEN}🎮 [GAME ENDED] Final cleanup | Agent: {self.animal_name}{Style.RESET_ALL}"
            )
            
            # Final cleanup
            self._enhanced_manager_cleanup()  # Use enhanced instead of nuclear for final cleanup
            
            # Stop monitoring
            self.system_monitor.stop_monitoring()
            
        except Exception as e:
            get_logger().error(f"Game end cleanup failed: {e}")

    def _save_to_hf(self):
        """Save model to HuggingFace with error handling"""
        if (
            self.hf_token not in [None, "None"]
            and self.state.round % self.hf_push_frequency == 0
        ):
            get_logger().info(
                f"{Fore.BLUE}📤 [HF PUSH] Pushing model to Hugging Face Hub | Round: {self.state.round}{Style.RESET_ALL}"
            )
            try:
                repo_id = self.trainer.args.hub_model_id
                if repo_id is None:
                    repo_id = Path(self.trainer.args.output_dir).name

                self.trainer.model.push_to_hub(
                    repo_id=repo_id,
                    token=self.hf_token,
                    commit_message=f"rl-swarm: round {self.state.round}, agent {self.animal_name}",
                    tags=[
                        "rl-swarm", "genrl-swarm", "grpo", "gensyn",
                        f"I am {self.animal_name}",
                    ],
                )
                
                get_logger().info(
                    f"{Fore.GREEN}✅ [HF SUCCESS] Model pushed successfully to {repo_id}{Style.RESET_ALL}"
                )
                
                # Cleanup after HF push
                gc.collect()
                
            except Exception as e:
                get_logger().error(f"{Fore.RED}❌ [HF FAILED] Failed to push model: {str(e)}{Style.RESET_ALL}")

    def agent_block(self, check_interval=5.0, log_timeout=10.0, max_check_interval=60.0 * 15):
        """Agent blocking with memory management"""
        start_time = time.monotonic()
        fetch_log_time = start_time
        check_backoff = check_interval
        
        self.agent_block_counter += 1
        
        get_logger().info(
            f"{Fore.YELLOW}⏸️ [BLOCKING] Waiting for swarm round advancement... | "
            f"Agent: {self.animal_name}{Style.RESET_ALL}"
        )
        
        while time.monotonic() - start_time < self.train_timeout and self.training_active:
            curr_time = time.monotonic()
            
            try:
                _ = self.communication.dht.get_visible_maddrs(latest=True)
                round_num, stage = self.coordinator.get_round_and_stage()
            except Exception as e:
                if curr_time - fetch_log_time > log_timeout:
                    get_logger().debug(
                        f"{Fore.YELLOW}🔍 Could not fetch round and stage: {e}. "
                        f"Next check in {check_interval}s.{Style.RESET_ALL}"
                    )
                    fetch_log_time = curr_time
                time.sleep(check_interval)
                continue

            if round_num >= self.state.round:
                get_logger().info(
                    f"{Fore.GREEN}🐝 [JOINING] Joining round: {round_num} | "
                    f"Model: {self.model_display_name}{Style.RESET_ALL}"
                )
                check_backoff = check_interval
                self.state.round = round_num
                
                # Light cleanup before returning - Less frequent
                if self.agent_block_counter % 100 == 0:  # Every 100 blocks instead of 50
                    gc.collect()
                    get_logger().debug(
                        f"{Fore.CYAN}🧹 Agent block cleanup #{self.agent_block_counter}{Style.RESET_ALL}"
                    )
                    
                return
            else:
                get_logger().info(
                    f"{Fore.YELLOW}⏭️ Already finished round: {round_num}. "
                    f"Next check in {check_backoff}s.{Style.RESET_ALL}"
                )
                time.sleep(check_backoff)
                check_backoff = min(check_backoff * 2, max_check_interval)

            if round_num == self.max_round - 1:
                get_logger().info(
                    f"{Fore.MAGENTA}🏁 [FINAL ROUND] Reached maximum round: {self.max_round}{Style.RESET_ALL}"
                )
                return

        get_logger().info(
            f"{Fore.RED}⏰ [TIMEOUT] Training timed out after {self.train_timeout}s!{Style.RESET_ALL}"
        )

    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            if hasattr(self, 'system_monitor'):
                self.system_monitor.stop_monitoring()
            self._enhanced_manager_cleanup()  # Use enhanced instead of nuclear for destructor
        except:
            pass
