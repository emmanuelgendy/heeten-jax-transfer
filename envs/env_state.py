import jax.numpy as jnp
import equinox as eqx

class ComplexHemsConfig(eqx.Module):
    """Static configuration parameters for the thermodynamic environment."""
    # Episode params
    max_steps: int = 24
    time_step_duration_hours: float = 1.0
    time_step_duration_sec: float = 3600.0
    
    # Reward Weights
    cost_weight: float = 1.0
    comfort_weight: float = 5.0
    
    # Comfort Setpoints
    temp_setpoint_low: float = 20.0
    temp_setpoint_high: float = 24.0
    initial_indoor_temp: float = 22.0
    temp_clip_low: float = 15.0 
    temp_clip_high: float = 30.0
    
    # Building Thermal Model (ISO 13790)
    heat_mass_capacity: float = 1.65e7
    heat_transmission: float = 200.0
    
    # Battery Parameters
    battery_capacity_kwh: float = 10.0
    max_batt_charge_kw: float = 5.0
    max_batt_discharge_kw: float = 5.0
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    initial_soc: float = 0.5
    
    # HVAC Parameters
    hvac_max_heat_power: float = 5.0
    hvac_max_cool_power: float = -5.0
    hvac_cop_heat: float = 3.5
    hvac_cop_cool: float = 3.0
    
    # Normalization Bounds
    max_pv_norm: float = 15.0
    max_load_norm: float = 5.0
    max_price_norm: float = 0.50
    norm_temp_min: float = -5.0
    norm_temp_max: float = 35.0

class EnvState(eqx.Module):
    """The dynamic environment state that updates at every timestep."""
    time_idx: jnp.ndarray       # Current hour in the 24h episode (0 to 23)
    building_idx: jnp.ndarray   # Which of the 5 Heeten houses this env is running
    day_start_idx: jnp.ndarray  # The starting index in the full historical timeline
    battery_soc: jnp.ndarray    # Current battery state of charge (0.0 to 1.0)
    indoor_temp: jnp.ndarray    # Current indoor temperature (Celsius)