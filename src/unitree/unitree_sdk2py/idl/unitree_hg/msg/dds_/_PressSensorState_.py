"""
  Generated by Eclipse Cyclone DDS idlc Python Backend
  Cyclone DDS IDL version: v0.11.0
  Module: unitree_hg.msg.dds_
  IDL file: PressSensorState_.idl

"""

from dataclasses import dataclass

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types

# root module import for resolving types
# import unitree_hg


@dataclass
@annotate.final
@annotate.autoid("sequential")
class PressSensorState_(
    idl.IdlStruct, typename="unitree_hg.msg.dds_.PressSensorState_"
):
    pressure: types.array[types.float32, 12]
    temperature: types.array[types.float32, 12]
    lost: types.uint32
    reserve: types.uint32
