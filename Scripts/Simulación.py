"""example.py
An example of creating a simulator and processing the sensor outputs.
"""
# lib
import os
import time
import signal
import threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import multiprocessing
import numpy as np
import copy
from queue import Queue

# src


#from ODT2 import TRACK_ONE_IMG
import sys
sys.path.append(r'C:\Users\Beast\Desktop\Fiorella-monodrive-python-client\Algorithm2\examples\ODT2\ODT2')
#sys.path.append(r'C:\Users\Beast\Desktop\Fiorella-monodrive-python-client\Algorithm2\monodrive')
from monodrive.simulator import Simulator
from monodrive.sensors import *
from examples import TRACKONEFRAMEYOLO
import mainAlg2



# constants
VERBOSE = False
DISPLAY = True

# global
lock = threading.RLock()
processing = 0
running = True
camera_frame = None
state_frame = None
speed = None
acele = None

speedQueue = []
aceleQueue = []
brakeQueue = []
stepQueue = []

DataQueue = Queue()
BrakingQueue = Queue()

def desicionMaking():
    global speed

    i = 0
    while True:
        if not DataQueue.empty():
            cola = DataQueue.get()
            brake = mainAlg2.funPrincipal(i, cola, 0.05, speed)
            BrakingQueue.put(brake)
            i = i + 1



def camera_on_update(frame: CameraFrame) -> None:
    """Callback for gather radar data.

    Args:
        frame(CameraFrame): New radar data
    """
    with lock:
        global processing, camera_frame
        camera_frame = frame
        processing -= 1


def state_on_update(frame: StateFrame) -> None:
    """Callback for gather radar data.

    Args:
        frame(CameraFrame): New radar data
    """
    with lock:
        global processing, state_frame, speed, acele
        state_frame = frame
        speedC = copy.deepcopy(state_frame)
        speedcm = speedC.frame.vehicles[-1].state.odometry.linear_velocity.x # Esta en cm/s
        speed = speedcm/100 # Esta en m/s
        acele = speedC.frame.vehicles[-1].state.odometry.linear_acceleration.x # Esta en cm/s2
        processing -= 1
        


def perception_and_control1(i):
    # TODO, process sensor data and determine control values to send to ego

    while i > 1:
        if not BrakingQueue.empty():
            brake = BrakingQueue.get()
            return 0.5, 0, brake, 1  # fwd, right, brake, mode
    return 0.5, 0, 0, 1


def main():
    """
    main driver function
    """
    root = os.path.dirname(__file__)
    #paths = "/home/avsimulation/Documents/ADAS-ECU Simulation/monodrive-python-client-fixed_step_python_client/examples"

    # Flag to allow user to stop the simulation from SIGINT
    global running
    global state_frame
    global speed, acele
    global speedQueue, aceleQueue, brakeQueue, stepQueue

    # Construct simulator from file
    simulator = Simulator.from_file(
        os.path.join(root, 'configurations', 'simulator_fixed_step.json'),
        scenario=os.path.join(root, 'scenarios', 'changeLane.json'),
        sensors=os.path.join(root, 'configurations', 'sensors_camera.json'),
        weather=os.path.join(root, 'configurations', 'weather.json'),
        verbose=VERBOSE
    )
    

    # Start the simulation
    simulator.start()
    print('Starting simulator')
    try:
        # Subscribe to sensors of interest
        simulator.subscribe_to_sensor('Camera_8000', camera_on_update)
        simulator.subscribe_to_sensor('State_8700', state_on_update)

        if DISPLAY:

            fig = plt.figure('perception system', figsize=(5, 5))
            ax_camera = fig.add_subplot(1, 1, 1)
            ax_camera.set_axis_off()

            fig.canvas.draw()
            data_camera = None


        # Start stepping the simulator
        time_steps = []
        i = 0

        while running and i < 10:
            start_time = time.time()

            with lock:
                global processing
                processing = simulator.active_sensor_count()

            # compute and send vehicle control command
            new_frame = copy.deepcopy(state_frame)

            if VERBOSE:
                print("sending frame: ", new_frame)

            
            if new_frame is not None:
                diU = time.time()
                response = simulator.update_state(new_frame.serialize(), 0.05)
                dfU = time.time()
                print("Tiempo update State: ", dfU - diU)

            if VERBOSE:
                print(response)

            # compute and send vehicle control command
            diDM = time.time()
            forward, right, brake, drive_mode = perception_and_control1(i)
            dfDM = time.time()
            print("Tiempo control: ", dfDM - diDM)
            print("Step: ", i)


            #print("Frenando ", brake)
            #print("Speed ", speed) # Para obtener la velocidad en m/s hay que dividir entre 100
            #print("Acceleration ", acele)

            #if i > 0:
                #stepQueue.append(i)
                #speedQueue.append(speed)
                #aceleQueue.append(acele)
                #brakeQueue.append(brake)


            if VERBOSE:
                print("sending control: {0}, {1}, {2}, {3}".format(
                    forward, right, brake, drive_mode
                ))

            diSS = time.time()
            response = simulator.send_control(forward, right, brake, drive_mode)
            dfSS = time.time()
            print("Tiempo para enviar Send control: ", dfSS - diSS)
            
            if VERBOSE:
                print(response)

            diSam = time.time()
            response = simulator.sample_sensors()
            dfSam = time.time()
            print("Tiempo sample sensor: ", dfSam - diSam)

            if VERBOSE:
                print(response)

            # wait for processing to complete
            while running:
                with lock:

                    if processing == 0:
                        break
                time.sleep(0.05)

            # plot if needed
            if DISPLAY:
                global camera_frame
                # update with camera data
                if camera_frame:
                    diI = time.time()
                    im = np.squeeze(camera_frame.image[..., ::-1])
                    dfI = time.time()
                    print("Tiempo de la imagen: ", dfI-diI)

                    diT = time.time()
                    nuevalista = TRACKONEFRAMEYOLO.detect_track_ODT(im)
                    dfT = time.time()
                    print("Tiempo ODT: ", dfT - diT)
                    #print("Nueva lista", nuevalista)
                    DataQueue.put(nuevalista)


                    if data_camera is None:
                        data_camera = ax_camera.imshow(im)
                    else:
                        data_camera.set_data(im)

                # do draw
                fig.canvas.draw()
                fig.canvas.flush_events()
                #plt.pause(0.0001)


            # timing
            print("--------------------------------------------------------")
            dt = time.time() - start_time
            time_steps.append(dt)
            if VERBOSE:
                print("Step = {0} completed in {1:.2f}ms".format(len(time_steps), (dt * 1000), 2))
                print("------------------")
            if running is False:
                break

            i = i + 1
        

        fps = 1.0 / (sum(time_steps) / len(time_steps))
        print('Average FPS: {}'.format(fps))
    except Exception as e:
        print("Exception thrown while running:", e)

    print("Stopping the simulator.")
    simulator.stop()


if __name__ == "__main__":
    #main()
    t1 = threading.Thread(target=main).start()
    t2 = threading.Thread(target=desicionMaking).start()
