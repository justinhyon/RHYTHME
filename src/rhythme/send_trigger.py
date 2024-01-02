import u3
import random
# import sg33500b as sg
import time
import serial
import serial.tools.list_ports
# labjackU3.configure()
def find_arduino(port=None):
    """Get the name of the port that is connected to Arduino."""
    if port is None:
        ports = serial.tools.list_ports.comports()
        for p in ports:
            if p.manufacturer is not None and "Arduino" in p.manufacturer:
                port = p.device
    return port


def handshake_arduino(
    arduino, sleep_time=1, print_handshake_message=False, handshake_code=0
):
    """Make sure connection is established by sending
    and receiving bytes."""
    # Close and reopen
    arduino.close()
    arduino.open()

    # Chill out while everything gets set
    time.sleep(sleep_time)

    # Set a long timeout to complete handshake
    timeout = arduino.timeout
    arduino.timeout = 2

    # Read and discard everything that may be in the input buffer
    _ = arduino.read_all()

    # Send request to Arduino
    arduino.write(bytes([handshake_code]))

    # Read in what Arduino sent
    handshake_message = arduino.read_until()

    # Send and receive request again
    arduino.write(bytes([handshake_code]))
    handshake_message = arduino.read_until()

    # Print the handshake message, if desired
    if print_handshake_message:
        print("Handshake message: " + handshake_message.decode())

    # Reset the timeout
    arduino.timeout = timeout

if __name__ == "__main__":
    HANDSHAKE = 0
    OFF = 2
    SIGNAL = 3

    port = find_arduino()
    arduino = serial.Serial(port, baudrate=115200)
    handshake_arduino(arduino, handshake_code=HANDSHAKE, print_handshake_message=True)

    # def send_trigger():
    d = u3.U3()
    d.configU3(FIOAnalog = 0, FIODirection = 255, FIOState = 0)
    d.configIO()

    # s = u3.U3()
    # s.configU3(FIOAnalog = 5, FIODirection = 255, FIOState = 0)
    # s.configIO()

    # set triggers for different conditions
    baseline_voltage = 0
    baseline_trigger = 3
    # s_trig = 7
    d.getFeedback(u3.PortStateWrite(State=[baseline_voltage, 0x00, 0x00], WriteMask=[0xff, 0x00, 0x00]))

    while True:
    # make sure that it is at baseline 0 beforehand:

        # s.getFeedback( u3.PortStateWrite(State = [baseline_voltage, 0x00, 0x00], WriteMask = [0xff, 0x00, 0x00] ) )

        #apply trigger (not baseline voltage) at the beginning of each trial
        print("trigger value ", baseline_trigger)

        d.getFeedback( u3.PortStateWrite(State = [baseline_trigger, 0x00, 0x00], WriteMask = [0xff, 0x00, 0x00] ) )

        arduino.write(bytes([SIGNAL]))

        # time.sleep(.2)
        # s.getFeedback( u3.PortStateWrite(State = [s_trig, 0x00, 0x00], WriteMask = [0xff, 0x00, 0x00] ) )
        # time.sleep(.5) # this should be removed when testing ERPs (the time sleep here was for testing in a Python IDE (e.g. Spyder) to see if it works

        #reset back to 0:
        d.getFeedback( u3.PortStateWrite(State = [baseline_voltage, 0x00, 0x00], WriteMask = [0xff, 0x00, 0x00] ) )
        # s.getFeedback( u3.PortStateWrite(State = [baseline_voltage, 0x00, 0x00], WriteMask = [0xff, 0x00, 0x00] ) )

        time.sleep(5)