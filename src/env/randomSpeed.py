import numpy as np

class randomSpeedProfile:
    """
    this class allows to integrate speed ramps during training,
    utilization of these speed ramps did not seem expedient in some initial tests,
    that is why the changeProbability (probability of a ramp occuring) was set to zero for this setup

    accordingly, this function is only responsible for changing the speed for every episode,
    during the episode the speed is kept constant
    """

    def __init__(self, epsLength, maxSpeed, changeProbability=1/20000, rampDuration=1.0):
        self.epsLength = epsLength
        self.maxSpeed = maxSpeed
        self.changeProbability = changeProbability
        self.rampDuration = rampDuration

        self.now_speed = 0.0 #np.random.uniform(-1, 1)
        self.next_speed = self.now_speed
        self.old_speed = self.now_speed
        self.upcoming_speed = self.now_speed

    def randomProfile(self, t):

        if (t <= 50e-6):
            self.now_speed = self.upcoming_speed
            self.next_speed = self.now_speed
            self.old_speed = self.now_speed
            self.t_ramp_end = 0
            self.upcoming_speed = 0.0#np.random.uniform(-1, 1)

        if self.old_speed == self.next_speed and np.random.uniform() < self.changeProbability:
            self.old_speed = self.next_speed
            self.next_speed = np.random.uniform(-1, 1)
            self.t_ramp_start = t
            self.t_ramp_end = t + self.rampDuration

        if t < self.t_ramp_end:
            self.now_speed = (t - self.t_ramp_start) * (self.next_speed - self.old_speed) / self.rampDuration + self.old_speed
        else:
            self.old_speed = self.next_speed

        return self.now_speed * self.maxSpeed