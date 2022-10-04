import u3
import random
# import sg33500b as sg
import time
# labjackU3.configure()

# def send_trigger():
d = u3.U3()
d.configU3(FIOAnalog = 0, FIODirection = 255, FIOState = 0)
d.configIO()

s = u3.U3()
s.configU3(FIOAnalog = 5, FIODirection = 255, FIOState = 0)
s.configIO()

# set triggers for different conditions
baseline_voltage = 0
baseline_trigger = 3

s_trig = 7

while True:
# make sure that it is at baseline 0 beforehand:
    d.getFeedback( u3.PortStateWrite(State = [baseline_voltage, 0x00, 0x00], WriteMask = [0xff, 0x00, 0x00] ) )
    s.getFeedback( u3.PortStateWrite(State = [baseline_voltage, 0x00, 0x00], WriteMask = [0xff, 0x00, 0x00] ) )

    #apply trigger (not baseline voltage) at the beginning of each trial
    d.getFeedback( u3.PortStateWrite(State = [baseline_trigger, 0x00, 0x00], WriteMask = [0xff, 0x00, 0x00] ) )
    time.sleep(.2)
    s.getFeedback( u3.PortStateWrite(State = [s_trig, 0x00, 0x00], WriteMask = [0xff, 0x00, 0x00] ) )
    time.sleep(.5) # this should be removed when testing ERPs (the time sleep here was for testing in a Python IDE (e.g. Spyder) to see if it works

    #reset back to 0:
    d.getFeedback( u3.PortStateWrite(State = [baseline_voltage, 0x00, 0x00], WriteMask = [0xff, 0x00, 0x00] ) )
    s.getFeedback( u3.PortStateWrite(State = [baseline_voltage, 0x00, 0x00], WriteMask = [0xff, 0x00, 0x00] ) )

    time.sleep(1)