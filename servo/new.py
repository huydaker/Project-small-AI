import time
from adafruit_servokit import ServoKit

# Khởi tạo đối tượng ServoKit với địa chỉ I2C của PCA9685 và số kênh servo
kit = ServoKit(channels=16)

# Thiết lập góc quay tối thiểu và tối đa của servo
up, down = 1, 0
a = 0
while a < 180:
    kit.servo[up].angle = a * 0.8
    kit.servo[down].angle = a
    a += 1
    time.sleep(0.001)
    

time.sleep(1)
kit.servo[up].angle = 90
kit.servo[down].angle = 90

# class see():
    # def __init__(self, up, down):
# 

# Hàm để chuyển đổi giá trị góc thành giá trị P
# i = 14
# kit.servo[0].angle = i
# kit.servo[1].angle = i
# for i in range (14, 167, 1):
#     kit.servo[0].angle = i
#     kit.servo[1].angle = i
# time.sleep(1)
# kit.servo[0].angle = 90
# kit.servo[1].angle = 90
# time.sleep(1)
# kit.servo[0].angle = 14
# kit.servo[1].angle = 14
# time.sleep(1)
# kit.servo[0].angle = 90
# kit.servo[1].angle = 90


    # i += 1
# a = 167
# for a in range (14, 167, -1):
#     kit.servo[0].angle = i
#     kit.servo[1].angle = i
#     time.sleep(2)



# kit.servo[0].angle = 180
# time.sleep(0.5)
# kit.servo[0].angle = 1
# time.sleep(0.5)
# kit.servo[0].angle = 111
# time.sleep(0.5)

# a = 1
# while a < 10:
#     kit.servo[0].angle = 0
#     time.sleep(3)
#     kit.servo[0].angle = 180
#     time.sleep(0.1)
#     a+=1
