class DummyC9:
    def set_pump_speed(self, *args, **kwargs): pass
    def close_clamp(self): pass
    def move_z(self, *args, **kwargs): pass
class North_Robot:
    def __init__(self, *args, **kwargs):
        self.c9 = DummyC9()
    def set_pipet_tip_type(self, *args, **kwargs): pass
    def move_vial_to_clamp(self, *args, **kwargs): pass
    def uncap_clamp_vial(self, *args, **kwargs): pass
    def aspirate_from_vial(self, *args, **kwargs): pass
    def dispense_into_wellplate(self, *args, **kwargs): pass
    def remove_pipet(self, *args, **kwargs): pass
    def recap_clamp_vial(self, *args, **kwargs): pass
    def return_vial_from_clamp(self, *args, **kwargs): pass
    def set_robot_speed(self, *args, **kwargs): pass
