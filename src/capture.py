import os
import subprocess
import time

def capture_dark():
    subprocess.run(["gphoto2", "--auto-detect"])
    for i in range(1, 51):
        subprocess.run(['gphoto2', '--set-config-value', '/main/capturesettings/shutterspeed=1/6'])
        subprocess.run(['gphoto2', '--capture-image-and-download', '--filename', f'../data/dark_frames/exposure{i}.%C'])

def capture_hdr():
    time.sleep(10)
    subprocess.run(["gphoto2", "--auto-detect"])
    
    subprocess.run(['gphoto2', '--set-config-value', '/main/capturesettings/shutterspeed=1/2048'])
    subprocess.run(['gphoto2', '--capture-image-and-download', '--filename', f'../data/my_stack/exposure1.%C'])

    subprocess.run(['gphoto2', '--set-config-value', '/main/capturesettings/shutterspeed=1/1024'])
    subprocess.run(['gphoto2', '--capture-image-and-download', '--filename', f'../data/my_stack/exposure2.%C'])
    
    subprocess.run(['gphoto2', '--set-config-value', '/main/capturesettings/shutterspeed=1/512'])
    subprocess.run(['gphoto2', '--capture-image-and-download', '--filename', f'../data/my_stack/exposure3.%C'])
    
    subprocess.run(['gphoto2', '--set-config-value', '/main/capturesettings/shutterspeed=1/256'])
    subprocess.run(['gphoto2', '--capture-image-and-download', '--filename', f'../data/my_stack/exposure4.%C'])
    
    subprocess.run(['gphoto2', '--set-config-value', '/main/capturesettings/shutterspeed=1/128'])
    subprocess.run(['gphoto2', '--capture-image-and-download', '--filename', f'../data/my_stack/exposure5.%C'])
    
    subprocess.run(['gphoto2', '--set-config-value', '/main/capturesettings/shutterspeed=1/64'])
    subprocess.run(['gphoto2', '--capture-image-and-download', '--filename', f'../data/my_stack/exposure6.%C'])
    
    subprocess.run(['gphoto2', '--set-config-value', '/main/capturesettings/shutterspeed=1/32'])
    subprocess.run(['gphoto2', '--capture-image-and-download', '--filename', f'../data/my_stack/exposure7.%C'])
    
    subprocess.run(['gphoto2', '--set-config-value', '/main/capturesettings/shutterspeed=1/16'])
    subprocess.run(['gphoto2', '--capture-image-and-download', '--filename', f'../data/my_stack/exposure8.%C'])
    
    subprocess.run(['gphoto2', '--set-config-value', '/main/capturesettings/shutterspeed=1/8'])
    subprocess.run(['gphoto2', '--capture-image-and-download', '--filename', f'../data/my_stack/exposure9.%C'])
    
    subprocess.run(['gphoto2', '--set-config-value', '/main/capturesettings/shutterspeed=1/4'])
    subprocess.run(['gphoto2', '--capture-image-and-download', '--filename', f'../data/my_stack/exposure10.%C'])
    
    subprocess.run(['gphoto2', '--set-config-value', '/main/capturesettings/shutterspeed=1/2'])
    subprocess.run(['gphoto2', '--capture-image-and-download', '--filename', f'../data/my_stack/exposure11.%C'])
    
    subprocess.run(['gphoto2', '--set-config-value', '/main/capturesettings/shutterspeed=1'])
    subprocess.run(['gphoto2', '--capture-image-and-download', '--filename', f'../data/my_stack/exposure12.%C'])
    
    subprocess.run(['gphoto2', '--set-config-value', '/main/capturesettings/shutterspeed=2'])
    subprocess.run(['gphoto2', '--capture-image-and-download', '--filename', f'../data/my_stack/exposure13.%C'])
    
    subprocess.run(['gphoto2', '--set-config-value', '/main/capturesettings/shutterspeed=4'])
    subprocess.run(['gphoto2', '--capture-image-and-download', '--filename', f'../data/my_stack/exposure14.%C'])
    
    subprocess.run(['gphoto2', '--set-config-value', '/main/capturesettings/shutterspeed=8'])
    subprocess.run(['gphoto2', '--capture-image-and-download', '--filename', f'../data/my_stack/exposure15.%C'])
    
    subprocess.run(['gphoto2', '--set-config-value', '/main/capturesettings/shutterspeed=16'])
    subprocess.run(['gphoto2', '--capture-image-and-download', '--filename', f'../data/my_stack/exposure16.%C'])

if __name__ == '__main__':
    capture_dark()