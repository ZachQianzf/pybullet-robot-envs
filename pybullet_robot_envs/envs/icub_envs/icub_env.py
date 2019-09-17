import os,  inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)

import pybullet as p
import robot_data
import math as m

class iCubEnv:

    def __init__(self, urdfRootPath=robot_data.getDataPath(),
                    timeStep=0.01,
                    useInverseKinematics=0, arm='l', useOrientation=0):

        self.urdfRootPath = os.path.join(urdfRootPath, "iCub/icub_fixed_model.sdf")
        self.timeStep = timeStep
        self.useInverseKinematics = useInverseKinematics
        self.useOrientation = useOrientation
        self.useSimulation = 1

        self.indices_torso = range(12, 15)
        self.indices_left_arm = range(15, 22)
        self.indices_right_arm = range(25, 32)
        self.indices_head = range(22, 25)

        self.home_pos_torso = [0.0, 0.0, 0.0] #degrees
        self.home_pos_head = [0.47, 0, 0]

        self.home_left_arm = [-29.4, 40.0, 0, 70, 0, 0, 0]
        self.home_right_arm = [-29.4, 40.0, 0, 70, 0, 0, 0]
        self.hand_pose = []

        self.workspace_lim = [[0.2, 0.52], [-0.2, 0.2], [0.5, 1.0]]

        self.control_arm = arm if arm =='r' or arm =='l' else 'l' #left arm by default

        self.reset()

    def reset(self):
        self.icubId = p.loadSDF(self.urdfRootPath)[0]
        self.numJoints = p.getNumJoints(self.icubId)

        # set constraint between base_link and world
        self.constr_id = p.createConstraint(self.icubId, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0],
                                 [p.getBasePositionAndOrientation(self.icubId)[0][0],
                                  p.getBasePositionAndOrientation(self.icubId)[0][1],
                                  p.getBasePositionAndOrientation(self.icubId)[0][2]],
                                  p.getBasePositionAndOrientation(self.icubId)[1])

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
            return 6 #position x,y,z + roll/pitch/yaw of hand frame
        return 3 #position x,y,z

    def getObservationDimension(self):
        return len(self.getObservation())

    def getObservation(self):
        # Cartesian world pos/orn of hand center of mass
        observation = []
        state = p.getLinkState(self.icubId, self.motorIndices[-1], computeLinkVelocity=1)
        pos = state[0]
        orn = p.getEulerFromQuaternion(state[1])
        if not self.hand_pose:
            self.hand_pose[:3] = pos
            self.hand_pose[3:6] = orn

        velL = state[6]
        velA = state[7]

        observation.extend(list(pos))
        observation.extend(list(orn)) #roll, pitch, yaw
        observation.extend(list(velL))
        observation.extend(list(velA))

        jointStates = p.getJointStates(self.icubId, self.motorIndices)
        jointPoses = [x[0] for x in jointStates]
        observation.extend(list(jointPoses))

        return observation

    def applyAction(self, action):

        if(self.useInverseKinematics):

            if not len(action) >= 3:
                raise AssertionError('number of action commands must be equal to 3 at least (dx,dy,dz)', len(action))

            dx, dy, dz = action[:3]

            self.hand_pose[0] = min(self.workspace_lim[0][1], max(self.workspace_lim[0][0], dx))
            self.hand_pose[1] = min(self.workspace_lim[1][1], max(self.workspace_lim[1][0], dy))
            self.hand_pose[2] = min(self.workspace_lim[2][1], max(self.workspace_lim[2][0], dz))

            if not self.useOrientation:
                quat_orn = p.getQuaternionFromEuler(self.handOrn)

            elif len(action) is 6:
                droll, dpitch, dyaw = action[3:]
                self.hand_pose[3] = min(m.pi, max(-m.pi, droll))
                self.hand_pose[4] = min(m.pi, max(-m.pi, dpitch))
                self.hand_pose[5] = min(m.pi, max(-m.pi, dyaw))

                quat_orn = p.getQuaternionFromEuler(self.hand_pose[3:6])

            else:
                quat_orn = p.getLinkState(self.icubId, self.motorIndices[-3])[5]

            # compute joint positions with IK
            jointPoses = p.calculateInverseKinematics(self.icubId, self.motorIndices[-1], self.hand_pose[:3], quat_orn)

            if self.useSimulation:
                for i in range(self.numJoints):
                    jointInfo = p.getJointInfo(self.icubId, i)
                    if jointInfo[3] > -1:
                        p.setJointMotorControl2(bodyUniqueId=self.icubId,
                                                jointIndex=i,
                                                controlMode=p.POSITION_CONTROL,
                                                targetPosition=jointPoses[i],
                                                targetVelocity=0,
                                                positionGain=0.25,
                                                velocityGain=0.75,
                                                force=50)
            else:
                # reset the joint state (ignoring all dynamics, not recommended to use during simulation)
                for i in range(self.numJoints):
                    p.resetJointState(self.icubId, i, jointPoses[i])

        else:
            if not len(action)==len(self.motorIndices):
                raise AssertionError('number of motor commands differs from number of motor to control', len(action),len(self.motorIndices))

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