import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_


def LowStateHandler(msg: LowState_):
    print(msg.motor_state)


ChannelFactoryInitialize(0, "enp2s0")
sub = ChannelSubscriber("rt/lowstate", LowState_)
sub.Init(LowStateHandler, 10)

while True:
    time.sleep(10.0)
