import os,  inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)

import pybullet as p
import robot_data

import numpy as np
import quaternion
import math as m

from utils import *

class iCubEnv:

    def __init__(self, urdfRootPath=robot_data.getDataPath(),
                    useInverseKinematics=0, arm='l', useOrientation=0):

        self.urdfRootPath = os.path.join(urdfRootPath, "iCub/icub_fixed_model.sdf")
        self.useInverseKinematics = useInverseKinematics
        self.useOrientation = useOrientation
        self.useSimulation = 1
        self._grasp_constr_id = -1

        self.indices_torso = range(12, 15)
        self.indices_left_arm = range(15, 22)
        self.indices_right_arm = range(25, 32)
        self.indices_head = range(22, 25)

        self.home_pos_torso = [0.0, 0.0, 0.0] #degrees
        self.home_pos_head = [0.47, 0, 0]

        self.home_left_arm = [-29.4, 40.0, 0, 70, 0, 0, 0]
        self.home_right_arm = [-29.4, 40.0, 0, 70, 0, 0, 0]

        self.workspace_lim = [[0.2, 0.52], [-0.2, 0.2], [0.5, 1.0]]

        self.control_arm = arm if arm =='r' or arm =='l' else 'l' #left arm by default

        self.reset()

    def reset(self):
        self.icubId = p.loadSDF(self.urdfRootPath)[0]
        self.numJoints = p.getNumJoints(self.icubId)

        # set constraint between base_link and world
        self.constr_id = p.createConstraint(self.icubId, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0],
                                            parentFramePosition=[0, 0, 0],
                                            childFramePosition=[p.getBasePositionAndOrientation(self.icubId)[0][0],
                                                                p.getBasePositionAndOrientation(self.icubId)[0][1],
                                                                p.getBasePositionAndOrientation(self.icubId)[0][2]*1.2],
                                            parentFrameOrientation=p.getBasePositionAndOrientation(self.icubId)[1])

        ## Set all joints initial values
        for count, i in enumerate(self.indices_torso):
            p.resetJointState(self.icubId, i, self.home_pos_torso[count]/180*m.pi)
            p.setJointMotorControl2(self.icubId, i, p.POSITION_CONTROL,
                targetPosition=self.home_pos_torso[count]/180*m.pi,
                targetVelocity=0.0, positionGain=0.25, velocityGain=0.75, force=50)

        for count, i in enumerate(self.indices_head):
            p.resetJointState(self.icubId, i, self.home_pos_head[count]/180*m.pi)
            p.setJointMotorControl2(self.icubId, i, p.POSITION_CONTROL,
                targetPosition=self.home_pos_head[count]/180*m.pi,
                targetVelocity=0.0, positionGain=0.25, velocityGain=0.75, force=50)

        for count, i in enumerate(self.indices_left_arm):
            p.resetJointState(self.icubId, i, self.home_left_arm[count]/180*m.pi)
            p.setJointMotorControl2(self.icubId, i, p.POSITION_CONTROL,
                targetPosition=self.home_left_arm[count]/180*m.pi,
                targetVelocity=0.0, positionGain=0.25, velocityGain=0.75, force=50)

        for count, i in enumerate(self.indices_right_arm):
            p.resetJointState(self.icubId, i, self.home_right_arm[count]/180*m.pi)
            p.setJointMotorControl2(self.icubId, i, p.POSITION_CONTROL,
                targetPosition=self.home_right_arm[count]/180*m.pi,
                targetVelocity=0.0, positionGain=0.25, velocityGain=0.75, force=50)

        self.ll, self.ul, self.jr, self.rs = self.getJointRanges()

        # save indices of only the joints to control
        control_arm_indices = self.indices_left_arm if self.control_arm == 'l' else self.indices_right_arm
        self.motorIndices = [i for i in self.indices_torso] + [j for j in control_arm_indices]

        self.motorNames = []
        for i in self.indices_torso:
            jointInfo = p.getJointInfo(self.icubId, i)
            if jointInfo[3] > -1:
                self.motorNames.append(str(jointInfo[1]))
        for i in control_arm_indices:
            jointInfo = p.getJointInfo(self.icubId, i)
            if jointInfo[3] > -1:
                self.motorNames.append(str(jointInfo[1]))

        self._grasp_constr_id = -1

    def getJointRanges(self):

        lowerLimits, upperLimits, jointRanges, restPoses = [], [], [], []
        for i in range(self.numJoints):
            jointInfo = p.getJointInfo(self.icubId, i)

            if jointInfo[3] > -1:
                ll, ul = jointInfo[8:10]
                jr = ul - ll
                # For simplicity, assume resting state == initial state
                rp = p.getJointState(self.icubId, i)[0]
                lowerLimits.append(ll)
                upperLimits.append(ul)
                jointRanges.append(jr)
                restPoses.append(rp)

        return lowerLimits, upperLimits, jointRanges, restPoses

    def getActionDimension(self):
        if not self.useInverseKinematics:
            return len(self.motorIndices)
        if self.useOrientation:
            return 7 #position x,y,z + roll/pitch/yaw of hand frame + open/close fingers
        return 3 #position x,y,z

    def getObservationDimension(self):
        return len(self.getObservation())

    def checkContactPalm(self):
        contacts = p.getContactPoints(bodyA=self.icubId, linkIndexA=self.motorIndices[-1])
        palm_pose = self.getObservation()[:6]
        count = 0
        for p_c in contacts:
            # pc_hand contact in hand COM frame coordinates
            h_T_w = p.invertTransform(palm_pose[:3], p.getQuaternionFromEuler(palm_pose[-3:]))
            h_R_w_matrix = p.getMatrixFromQuaternion(h_T_w[1])
            h_R_w = np.array([h_R_w_matrix[0:3], h_R_w_matrix[3:6], h_R_w_matrix[6:9]])
            pc_h = h_R_w.dot(p_c[5]) + h_T_w[0]
            if -0.04 <= pc_h[0] <= 0.02 and pc_h[1] < 0:
                count = count + 1
        return True if count >= 2 else False

    def releaseObject(self):
        if self.isGrasping():
            p.removeConstraint(self._grasp_constr_id)
            self._grasp_constr_id = -1
            return True
        return False

    def isGrasping(self):
        if self._grasp_constr_id is not -1:
            # constraint already created
            return True
        return False

    def graspObject(self, object_id, obj_pose):
        # return if it's already grasping
        if self.isGrasping():
            return True

        if not self.checkContactPalm():
            return False

        p_contact = p.getContactPoints(bodyA=self.icubId, bodyB=object_id,
                                       linkIndexA=self.motorIndices[-1])[0]
        w_pc_hand = p_contact[5]
        w_pc_obj = p_contact[6]

        palm_pose = self.getObservation()[:6]

        # pc_hand contact in hand COM frame coordinates
        h_T_w = p.invertTransform(palm_pose[:3], p.getQuaternionFromEuler(palm_pose[-3:]))
        h_R_w_matrix = p.getMatrixFromQuaternion(h_T_w[1])
        h_R_w = np.array([h_R_w_matrix[0:3], h_R_w_matrix[3:6], h_R_w_matrix[6:9]])
        pc_h = h_R_w.dot(w_pc_hand) + h_T_w[0]


        # pc_obj contact in obj COM frame coordinates
        obj_T_w = p.invertTransform(obj_pose[:3], p.getQuaternionFromEuler(obj_pose[-3:]))
        obj_R_w_matrix = p.getMatrixFromQuaternion(obj_T_w[1])
        obj_R_w = np.array([obj_R_w_matrix[0:3], obj_R_w_matrix[3:6], obj_R_w_matrix[6:9]])
        pc_obj = obj_R_w.dot(w_pc_obj) + obj_T_w[0]

        #hand quaternion
        w_q_h = np.array(p.getQuaternionFromEuler(self.getObservation()[3:6]))
        w_q_h = np.quaternion(w_q_h[-1], w_q_h[0], w_q_h[1], w_q_h[2])

        # obj quaternion
        w_q_obj = np.array(p.getQuaternionFromEuler(obj_pose[-3:]))
        w_q_obj = np.quaternion(w_q_obj[-1], w_q_obj[0], w_q_obj[1], w_q_obj[2])

        # relative quaternion from hand to object coord. frame
        h_q_obj = np.conj(w_q_h) * w_q_obj
        h_q_obj_a = quaternion.as_float_array(h_q_obj)

        self._grasp_constr_id = p.createConstraint(self.icubId, self.motorIndices[-1], object_id, -1, p.JOINT_FIXED, [0, 0, 0],
                                       parentFramePosition=pc_h,
                                       childFramePosition=pc_obj,
                                       parentFrameOrientation=[h_q_obj_a[1], h_q_obj_a[2], h_q_obj_a[3], h_q_obj_a[0]])
        return True

    def getObservation(self):
        # Cartesian world pos/orn of hand center of mass
        observation = []
        state = p.getLinkState(self.icubId, self.motorIndices[-1], computeLinkVelocity=1)
        pos = state[0]  # pos of COM in world coord
        orn = p.getEulerFromQuaternion(state[1])

        #velL = state[6]
        #velA = state[7]

        observation.extend(list(pos))
        observation.extend(list(orn)) #roll, pitch, yaw
        #observation.extend(list(velL))
        #observation.extend(list(velA))

        jointStates = p.getJointStates(self.icubId, self.motorIndices)
        jointPoses = [x[0] for x in jointStates]
        observation.extend(list(jointPoses))

        return observation

    def applyAction(self, action):

        if(self.useInverseKinematics):

            if not len(action) >= 3:
                raise AssertionError('number of action commands must be equal to 3 at least (dx,dy,dz)', len(action))

            curr_pose = self.getObservation()

            dx, dy, dz = action[:3]

            new_pos = [min(self.workspace_lim[0][1], max(self.workspace_lim[0][0], dx)),
                       min(self.workspace_lim[1][1], max(self.workspace_lim[1][0], dy)),
                       min(self.workspace_lim[2][1], max(self.workspace_lim[2][0], dz))]

            if not self.useOrientation:
                new_quat_orn = p.getQuaternionFromEuler(self.handOrn)

            elif len(action) >= 6:
                droll, dpitch, dyaw = action[3:6]
                new_eu_orn = [min(m.pi, max(-m.pi, droll)),
                              min(m.pi, max(-m.pi, dpitch)),
                              min(m.pi, max(-m.pi, dyaw))]

                new_quat_orn = p.getQuaternionFromEuler(new_eu_orn)

            else:
                new_quat_orn = p.getLinkState(self.icubId, self.motorIndices[-1])[1]

            # transform from com to link frame of hand

            # compute joint positions with IK
            jointPoses = list(p.calculateInverseKinematics(self.icubId, self.motorIndices[-1],
                                                      new_pos, new_quat_orn))

            # workaround to block joints of not-controlled arm
            joints_to_block = self.indices_left_arm if self.control_arm == 'r' else self.indices_right_arm

            if self.useSimulation:
                for i in range(self.numJoints):
                    if i in joints_to_block:
                        continue
                    jointInfo = p.getJointInfo(self.icubId, i)
                    if jointInfo[3] > -1:
                        # minimize error is:
                        # error = position_gain * (desired_position - actual_position) +
                        #         velocity_gain * (desired_velocity - actual_velocity)

                        p.setJointMotorControl2(bodyUniqueId=self.icubId,
                                                jointIndex=i,
                                                controlMode=p.POSITION_CONTROL,
                                                targetPosition=jointPoses[i],
                                                targetVelocity=0,
                                                maxVelocity=0.5,
                                                positionGain=0.25,
                                                velocityGain=1,
                                                force=50)
            else:
                # reset the joint state (ignoring all dynamics, not recommended to use during simulation)
                for i in range(self.numJoints):
                    if i in joints_to_block:
                        continue
                    p.resetJointState(self.icubId, i, jointPoses[i])

        else:
            if not len(action)==len(self.motorIndices):
                raise AssertionError('number of motor commands differs from number of motor to control', len(action), len(self.motorIndices))

            for idx,val in enumerate(action):
                motor = self.motorIndices[idx]

                curr_motor_pos = p.getJointState(self.icubId, motor)[0]
                new_motor_pos = min(self.ul[motor], max(self.ll[motor], curr_motor_pos + val))

                p.setJointMotorControl2(self.icubId,
                                        motor,
                                        p.POSITION_CONTROL,
                                        targetPosition=new_motor_pos,
                                        targetVelocity=0,
                                        positionGain=0.25,
                                        velocityGain=0.75,
                                        force=50)

    def debug_gui(self):

        ws = self.workspace_lim
        p1 = [ws[0][0], ws[1][0], ws[2][0]]  # xmin,ymin
        p2 = [ws[0][1], ws[1][0], ws[2][0]]  # xmax,ymin
        p3 = [ws[0][1], ws[1][1], ws[2][0]]  # xmax,ymax
        p4 = [ws[0][0], ws[1][1], ws[2][0]]  # xmin,ymax

        p.addUserDebugLine(p1, p2, lineColorRGB=[0, 0, 1], lineWidth=2.0, lifeTime=0)
        p.addUserDebugLine(p2, p3, lineColorRGB=[0, 0, 1], lineWidth=2.0, lifeTime=0)
        p.addUserDebugLine(p3, p4, lineColorRGB=[0, 0, 1], lineWidth=2.0, lifeTime=0)
        p.addUserDebugLine(p4, p1, lineColorRGB=[0, 0, 1], lineWidth=2.0, lifeTime=0)

        p.addUserDebugLine([0, 0, 0], [0.3, 0, 0], [1, 0, 0], parentObjectUniqueId=self.icubId, parentLinkIndex=-1)
        p.addUserDebugLine([0, 0, 0], [0, 0.3, 0], [0, 1, 0], parentObjectUniqueId=self.icubId, parentLinkIndex=-1)
        p.addUserDebugLine([0, 0, 0], [0, 0, 0.3], [0, 0, 1], parentObjectUniqueId=self.icubId, parentLinkIndex=-1)

        p.addUserDebugLine([0, 0, 0], [0.1, 0, 0], [1, 0, 0], parentObjectUniqueId=self.icubId, parentLinkIndex=self.indices_right_arm[-1])
        p.addUserDebugLine([0, 0, 0], [0, 0.1, 0], [0, 1, 0], parentObjectUniqueId=self.icubId, parentLinkIndex=self.indices_right_arm[-1])
        p.addUserDebugLine([0, 0, 0], [0, 0, 0.1], [0, 0, 1], parentObjectUniqueId=self.icubId, parentLinkIndex=self.indices_right_arm[-1])

        p.addUserDebugLine([0, 0, 0], [0.1, 0, 0], [1, 0, 0], parentObjectUniqueId=self.icubId, parentLinkIndex=self.indices_left_arm[-1])
        p.addUserDebugLine([0, 0, 0], [0, 0.1, 0], [0, 1, 0], parentObjectUniqueId=self.icubId, parentLinkIndex=self.indices_left_arm[-1])
        p.addUserDebugLine([0, 0, 0], [0, 0, 0.1], [0, 0, 1], parentObjectUniqueId=self.icubId, parentLinkIndex=self.indices_left_arm[-1])

        state = p.getLinkState(self.icubId, self.motorIndices[-1])[0]
        #state[0] = state[0]-
        #p.addUserDebugLine(state, state, lineColorRGB=[0, 0, 1], lineWidth=2.0, lifeTime=0)
        #p.addUserDebugLine(state, state, lineColorRGB=[0, 0, 1], lineWidth=2.0, lifeTime=0)
        #p.addUserDebugLine(state, state, lineColorRGB=[0, 0, 1], lineWidth=2.0, lifeTime=0)