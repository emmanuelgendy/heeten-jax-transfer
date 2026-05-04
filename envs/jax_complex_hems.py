import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from typing import Tuple

from .env_state import ComplexHemsConfig, EnvState

class JAXComplexHemsEnv(eqx.Module):
    config: ComplexHemsConfig
    
    # The loaded Heeten Tensors
    pv_data: jnp.ndarray          
    load_data: jnp.ndarray        
    outdoor_temp_data: jnp.ndarray 
    import_price_data: jnp.ndarray 
    export_price_data: jnp.ndarray 
    
    # Action Space Arrays
    num_actions: int
    action_batt_kw: jnp.ndarray
    action_hvac_kw: jnp.ndarray

    def __init__(self, data_path: str = "data/processed/heeten_complex_hems_data.npz"):
        self.config = ComplexHemsConfig()
        
        print(f"Loading Heeten Tensors from {data_path}...")
        data = np.load(data_path)
        self.pv_data = jnp.array(data['pv_data'])
        self.load_data = jnp.array(data['load_data'])
        self.outdoor_temp_data = jnp.array(data['outdoor_temp_data'])
        self.import_price_data = jnp.array(data['import_price_data'])
        self.export_price_data = jnp.array(data['export_price_data'])
        
        # Build the 9-Discrete Action Grid (Matches your PyTorch logic)
        batt_actions = [self.config.max_batt_discharge_kw, 0.0, self.config.max_batt_charge_kw]
        hvac_actions = [self.config.hvac_max_cool_power, 0.0, self.config.hvac_max_heat_power]
        
        batt_grid, hvac_grid = jnp.meshgrid(jnp.array(batt_actions), jnp.array(hvac_actions))
        self.action_batt_kw = batt_grid.flatten()
        self.action_hvac_kw = hvac_grid.flatten()
        self.num_actions = len(self.action_batt_kw)

    @jax.jit
    def reset(self, key: jax.Array) -> Tuple[jnp.ndarray, EnvState]:
        """Assigns a random Heeten building and random historical day."""
        k1, k2 = jax.random.split(key)
        
        num_buildings = self.pv_data.shape[0]
        building_idx = jax.random.randint(k1, shape=(), minval=0, maxval=num_buildings)
        
        max_valid_start = self.pv_data.shape[1] - self.config.max_steps
        day_start_idx = jax.random.randint(k2, shape=(), minval=0, maxval=max_valid_start)
        
        state = EnvState(
            time_idx=jnp.array(0, dtype=jnp.int32),
            building_idx=building_idx,
            day_start_idx=day_start_idx,
            battery_soc=jnp.array(self.config.initial_soc, dtype=jnp.float32),
            indoor_temp=jnp.array(self.config.initial_indoor_temp, dtype=jnp.float32)
        )
        return self._get_obs(state), state

    @jax.jit
    def step(self, state: EnvState, action_idx: jnp.ndarray) -> Tuple[jnp.ndarray, EnvState, jnp.ndarray, jnp.ndarray]:
        """Executes one thermodynamic timestep."""
        cfg = self.config
        global_t = state.day_start_idx + state.time_idx
        
        # 1. Fetch Environment Variables
        pv = self.pv_data[state.building_idx, global_t]
        load = self.load_data[state.building_idx, global_t]
        price_in = self.import_price_data[global_t]
        price_out = self.export_price_data[global_t]
        out_temp = self.outdoor_temp_data[global_t]
        
        # 2. Decode Actuator Commands
        target_batt = self.action_batt_kw[action_idx]
        target_hvac = self.action_hvac_kw[action_idx]
        
        # 3. Battery Thermodynamics
        e_to_full = (1.0 - state.battery_soc) * cfg.battery_capacity_kwh
        max_c = jnp.where(cfg.charge_efficiency > 0, (e_to_full / cfg.time_step_duration_hours) / cfg.charge_efficiency, jnp.inf)
        actual_max_c = jnp.minimum(cfg.max_batt_charge_kw, max_c)
        
        e_to_empty = state.battery_soc * cfg.battery_capacity_kwh
        max_d = (e_to_empty * cfg.discharge_efficiency) / cfg.time_step_duration_hours
        actual_max_d = jnp.minimum(cfg.max_batt_discharge_kw, max_d)
        
        actual_batt = jnp.where(target_batt > 0, jnp.minimum(target_batt, actual_max_c),
                      jnp.where(target_batt < 0, -jnp.minimum(jnp.abs(target_batt), actual_max_d), 0.0))
        
        e_change = jnp.where(actual_batt > 0, (actual_batt * cfg.time_step_duration_hours) * cfg.charge_efficiency,
                   jnp.where(actual_batt < 0, (actual_batt * cfg.time_step_duration_hours) / cfg.discharge_efficiency, 0.0))
        next_soc = jnp.clip(state.battery_soc + (e_change / cfg.battery_capacity_kwh), 0.0, 1.0)
        
        # 4. Building Thermal Dynamics
        thermal_power = jnp.where(target_hvac > 0, (target_hvac * 1000) * cfg.hvac_cop_heat,
                        jnp.where(target_hvac < 0, (target_hvac * 1000) * cfg.hvac_cop_cool, 0.0))
        
        thermal_loss = cfg.heat_transmission * (state.indoor_temp - out_temp)
        delta_temp = ((thermal_power - thermal_loss) * cfg.time_step_duration_sec) / cfg.heat_mass_capacity
        next_temp = jnp.clip(state.indoor_temp + delta_temp, cfg.temp_clip_low, cfg.temp_clip_high)
        
        # 5. Grid Power Flow
        net_power = pv - (load + jnp.abs(target_hvac)) - actual_batt
        grid_import = jnp.maximum(0.0, -net_power)
        grid_export = jnp.maximum(0.0, net_power)
        action_cost = (grid_import * price_in) - (grid_export * price_out)
        
        # Baseline Cost (Idle System)
        base_net = pv - load
        base_cost = (jnp.maximum(0.0, -base_net) * price_in) - (jnp.maximum(0.0, base_net) * price_out)
        
        # 6. Reward Calculation
        comfort_penalty = jnp.maximum(0.0, next_temp - cfg.temp_setpoint_high) + jnp.maximum(0.0, cfg.temp_setpoint_low - next_temp)
        
        cost_reward = (base_cost - action_cost) * cfg.cost_weight
        comfort_reward = -comfort_penalty * cfg.comfort_weight
        reward = cost_reward + comfort_reward
        
        # 7. State Progression
        next_state = EnvState(
            time_idx=state.time_idx + 1,
            building_idx=state.building_idx,
            day_start_idx=state.day_start_idx,
            battery_soc=next_soc,
            indoor_temp=next_temp
        )
        
        done = next_state.time_idx >= cfg.max_steps
        
        return self._get_obs(next_state), next_state, reward, done

    def _get_obs(self, state: EnvState) -> jnp.ndarray:
        cfg = self.config
        global_t = state.day_start_idx + state.time_idx
        
        norm_t = lambda t, min_t, max_t: (t - min_t) / (max_t - min_t)
        
        # 7-Dimensional Observation Vector
        obs = jnp.array([
            state.time_idx / cfg.max_steps,
            self.pv_data[state.building_idx, global_t] / cfg.max_pv_norm,
            self.load_data[state.building_idx, global_t] / cfg.max_load_norm,
            self.import_price_data[global_t] / cfg.max_price_norm,
            state.battery_soc,
            norm_t(state.indoor_temp, cfg.norm_temp_min, cfg.norm_temp_max),
            norm_t(self.outdoor_temp_data[global_t], cfg.norm_temp_min, cfg.norm_temp_max)
        ], dtype=jnp.float32)
        
        return obs