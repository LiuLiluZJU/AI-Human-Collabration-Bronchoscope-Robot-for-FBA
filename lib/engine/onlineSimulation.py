import os
import json
from cv2 import TM_CCOEFF_NORMED
from graphviz import render
import pybullet as p
import pybullet_data
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import cv2
from PIL import Image
import time
import datetime
import math
from collections import deque
from scipy.spatial.transform import Rotation
import numpy as np
import torch
from torchvision.transforms import Resize
import matplotlib.pyplot as plt
from mayavi import mlab
import trimesh
from pyrender import IntrinsicsCamera, PerspectiveCamera,\
                     DirectionalLight, SpotLight, PointLight,\
                     MetallicRoughnessMaterial,\
                     Primitive, Mesh, Node, Scene,\
                     Viewer, OffscreenRenderer, RenderFlags
import matplotlib.pyplot as plt
import networkx as nx
import pydot
from networkx.drawing.nx_pydot import graphviz_layout

from lib.engine.camera import fixedCamera
from lib.engine.keyBoardEvents import getAddition, getAdditionPlain, getDirection


def dcm2quat(R):
	
    epsilon = 1e-5
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    assert trace > -1
    if np.fabs(trace + 1) < epsilon:
        if np.argmax([R[0, 0], R[1, 1], R[2, 2]]) == 0:
            t = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
            q0 = (R[2, 1] - R[1, 2]) / t
            q1 = t / 4
            q2 = (R[0, 2] + R[2, 0]) / t
            q3 = (R[0, 1] + R[1, 0]) / t
        elif np.argmax([R[0, 0], R[1, 1], R[2, 2]]) == 1:
            t = np.sqrt(1 - R[0, 0] + R[1, 1] - R[2, 2])
            q0 = (R[0, 2] - R[2, 0]) / t
            q1 = (R[0, 1] + R[1, 0]) / t
            q2 = t / 4
            q3 = (R[2, 1] + R[1, 2]) / t
        else:
            t = np.sqrt(1 - R[0, 0] - R[1, 1] + R[2, 2])
            q0 = (R[1, 0] - R[0, 1]) / t
            q1 = (R[0, 2] + R[2, 0]) / t
            q2 = (R[1, 2] - R[2, 1]) / t
            q3 = t / 4
    else:
        q0 = np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2
        q1 = (R[2, 1] - R[1, 2]) / (4 * q0)
        q2 = (R[0, 2] - R[2, 0]) / (4 * q0)
        q3 = (R[1, 0] - R[0, 1]) / (4 * q0)

    return np.array([q1, q2, q3, q0])


class TreeNode(object):
    
    def __init__(self, val=None, father=None, id=None):
        self.val = val
        self.headIndex = None
        self.tailIndex = None
        self.vertexArray = None
        self.father = father
        self.id = id
        self.children = []


class ExploreTreeNode(object):
    
    def __init__(self, ref_img=None, condition_list=[], frontiers=[], father=None, id=None):
        self.ref_img = ref_img
        self.condition_list = condition_list
        self.frontiers = frontiers
        self.father = father
        self.id = id
        self.passed_trajectory = []
        self.children = []


class onlineSimulationWithNetwork(object):

    def __init__(self, args, centerline_name, renderer=None, training=True):

        # Load models
        name = centerline_name.split(" ")[0]
        self.bronchus_model_dir = os.path.join("Airways", "AirwayHollow_{}_simUV.obj".format(name))
        # self.bronchus_model_dir = os.path.join("Airways", "AirwayHollow_{}.obj".format(name))
        self.airway_model_dir = os.path.join("Airways", "AirwayModel_Peach_{}.vtk".format(name))
        network_model_dir = os.path.join("Airways", "Network_{}.obj".format(name))
        self.centerline_name = centerline_name
        centerline_model_name = centerline_name.lstrip(name + " ")
        self.centerline_model_dir = os.path.join("Airways", "centerline_models_{}".format(name), centerline_model_name + ".obj")

        p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(1. / 120.)
        # useMaximalCoordinates is much faster then the default reduced coordinates (Featherstone)
        p.loadURDF("plane100.urdf", useMaximalCoordinates=True)

        shift = [0, 0, 0]
        self.meshScale = [0.01, 0.01, 0.01]
        # meshScale = [0.0001, 0.0001, 0.0001]
        # the visual shape and collision shape can be re-used by all createMultiBody instances (instancing)
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,
                                            # fileName="C:/Users/leko/Downloads/AirwayModel_2_Peach.obj",
                                            fileName=self.bronchus_model_dir,
                                            rgbaColor=[249 / 255, 204 / 255, 226 / 255, 1],
                                            specularColor=[0, 0, 0],
                                            visualFramePosition=shift,
                                            meshScale=self.meshScale)
        collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                                # fileName="C:/Users/leko/Downloads/AirwayModel_2_Peach.obj",
                                                fileName=self.bronchus_model_dir,
                                                collisionFramePosition=shift,
                                                meshScale=self.meshScale)

        # Augment on roll angle
        if training:
            self.rand_roll = (np.random.rand() - 0.5) * 2 * np.pi
            # self.rand_roll = 0
        else:
            # self.rand_roll = (np.random.rand() - 0.5) * 2 * np.pi
            self.rand_roll = 0
        
        euler = p.getEulerFromQuaternion([np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2])
        self.quaternion_model = p.getQuaternionFromEuler([np.pi / 2, self.rand_roll, 0])
        # self.quaternion_model = p.getQuaternionFromEuler([0, self.rand_roll, 0])
        self.matrix_model = p.getMatrixFromQuaternion(self.quaternion_model)
        self.R_model = np.reshape(self.matrix_model, (3, 3))
        self.t_model = np.array([0, 0, 5])

        self.airwayBodyId = p.createMultiBody(baseMass=1,
                                            baseInertialFramePosition=[0, 0, 0],
                                            baseCollisionShapeIndex=collisionShapeId,
                                            baseVisualShapeIndex=visualShapeId,
                                            basePosition=[0, 0, 5],
                                            # baseOrientation=[np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2],
                                            baseOrientation=self.quaternion_model,
                                            #   basePosition=[0, 0, 10],
                                            useMaximalCoordinates=True)

        # p.setGravity(0, 0, -10)
        p.setRealTimeSimulation(1)

        # Set camera path
        file_path = self.centerline_model_dir
        reader = vtk.vtkOBJReader()
        reader.SetFileName(file_path)
        reader.Update()

        mesh = reader.GetOutput()
        points = mesh.GetPoints()
        data = points.GetData()
        centerlineArray = vtk_to_numpy(data)
        centerlineArray = np.dot(self.R_model, centerlineArray.T).T * 0.01 + self.t_model

        # Downsample or upsample the centerline to the same length/size rate
        centerline_length = 0
        for i in range(len(centerlineArray) - 1):
            length_diff = np.linalg.norm(centerlineArray[i] - centerlineArray[i + 1])
            centerline_length += length_diff
        centerline_size = len(centerlineArray)
        lenth_size_rate = 0.007  # refer to Siliconmodel1
        centerline_size_exp = int(centerline_length / lenth_size_rate)
        centerlineArray_exp = np.zeros((centerline_size_exp, 3))
        for index_exp in range(centerline_size_exp):
            index = index_exp / (centerline_size_exp - 1) * (centerline_size - 1)
            index_left_bound = int(index)
            index_right_bound = int(index) + 1
            if index_left_bound == centerline_size - 1:
                centerlineArray_exp[index_exp] = centerlineArray[index_left_bound]
            else:
                centerlineArray_exp[index_exp] = (index_right_bound - index) * centerlineArray[index_left_bound] + (index - index_left_bound) * centerlineArray[index_right_bound]
        centerlineArray = centerlineArray_exp

        # Smoothing trajectory
        self.originalCenterlineArray = centerlineArray
        centerlineArray_smoothed = np.zeros_like(centerlineArray)
        for i in range(len(centerlineArray)):
            left_bound = i - 10
            right_bound = i + 10
            # left_bound = i - 20
            # right_bound = i + 20
            if left_bound < 0: left_bound = 0
            if right_bound > len(centerlineArray): right_bound = len(centerlineArray)
            centerlineArray_smoothed[i] = np.mean(centerlineArray[left_bound : right_bound], axis=0)
        self.centerlineArray = centerlineArray_smoothed

        # Calculate trajectory length
        centerline_length = 0
        for i in range(len(self.centerlineArray) - 1):
            length_diff = np.linalg.norm(self.centerlineArray[i] - self.centerlineArray[i + 1])
            centerline_length += length_diff
        self.centerline_length = centerline_length

        # Generate new path in each step
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(self.airway_model_dir)
        reader.Update()
        self.vtkdata = reader.GetOutput()
        # self.centerlineExtractor = ExtractCenterline(self.vtkdata)
        self.targetPoint = centerlineArray[0]
        self.transformed_target = np.dot(np.linalg.inv(self.R_model), self.targetPoint - self.t_model) * 100
        self.transformed_target_vtk_cor = np.array([-self.transformed_target[0], -self.transformed_target[1], self.transformed_target[2]])  # x and y here is opposite to those in the world coordinate system

        # Collision detection
        self.pointLocator = vtk.vtkPointLocator()
        self.pointLocator.SetDataSet(self.vtkdata)
        self.pointLocator.BuildLocator()

        # Normal calculation
        normal_estimation = vtk.vtkPCANormalEstimation()
        normal_estimation.SetInputData(self.vtkdata)
        # normal_estimation.SetSampleSize(10)
        normal_estimation.SetNormalOrientationToGraphTraversal()
        normal_estimation.Update()
        self.normals = normal_estimation.GetOutput().GetPointData().GetNormals()
        # # 输出法线向量
        # for i in range(normals.GetNumberOfTuples()):
        #     normal = normals.GetTuple(i)
        #     print(f'Normal {i}: ({normal[0]}, {normal[1]}, {normal[2]})')

        self.camera = fixedCamera(0.01, p)
        # # camera.lookat(0, -89.999, [0.15, -0.05, -6])
        # camera.lookat(0, -90.001, [0, 0, 0])
        # # camera.getImg()
        # count = -6

        boundingbox = p.getAABB(self.airwayBodyId)
        print(boundingbox)
        print(np.max(centerlineArray, axis=0))
        print(np.min(centerlineArray, axis=0))
        print(np.argmax(centerlineArray, axis=0))
        # print(centerlineArray[1350])
        position = p.getBasePositionAndOrientation(self.airwayBodyId)

        # Pyrender initialization
        self.renderer = renderer
        fuze_trimesh = trimesh.load(self.bronchus_model_dir)
        # material = MetallicRoughnessMaterial(
        #                 metallicFactor=1.0,
        #                 alphaMode='OPAQUE',
        #                 roughnessFactor=0.7,
        #                 baseColorFactor=[253 / 255, 149 / 255, 158 / 255, 1])
        material = MetallicRoughnessMaterial(
                            metallicFactor=0.1,
                            alphaMode='OPAQUE',
                            roughnessFactor=0.7,
                            baseColorFactor=[206 / 255, 108 / 255, 131 / 255, 1])
        # self.fuze_mesh = Mesh.from_trimesh(fuze_trimesh, material=material)
        self.fuze_mesh = Mesh.from_trimesh(fuze_trimesh)
        spot_l = SpotLight(color=np.ones(3), intensity=0.3,
                        innerConeAngle=0, outerConeAngle=np.pi/2, range=1)
        # self.cam = IntrinsicsCamera(fx=181.9375, fy=183.2459, cx=103.0638, cy=95.4945, znear=0.000001)
        self.cam = IntrinsicsCamera(fx=175 / 1.008, fy=175 / 1.008, cx=200, cy=200, znear=0.00001)
        self.scene = Scene(bg_color=(0., 0., 0.))
        self.fuze_node = Node(mesh=self.fuze_mesh, scale=self.meshScale, rotation=self.quaternion_model, translation=self.t_model)
        self.scene.add_node(self.fuze_node)
        self.spot_l_node = self.scene.add(spot_l)
        self.cam_node = self.scene.add(self.cam)
        # self.r = OffscreenRenderer(viewport_width=200, viewport_height=200)
        self.r = OffscreenRenderer(viewport_width=400, viewport_height=400)

        for i in range(len(self.centerlineArray) - 1):
            p.addUserDebugLine(self.centerlineArray[i], self.centerlineArray[i + 1], lineColorRGB=[0, 1, 0], lifeTime=0, lineWidth=3)

        # Build tree
        networkVertex = []
        networkLineIndex = []
        lineLengthMax = 0
        line_count = 0
        lineLengthMaxCount = None
        with open(network_model_dir, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    _, x, y, z = line.split()
                    networkVertex.append([float(x), float(y), float(z)])
                elif line.startswith('l '):
                    _, *vertex_indices = line.split()
                    # print(len(vertex_indices))
                    lineIndex = []
                    lineLength = 0
                    for i, vertex_index in enumerate(vertex_indices):
                        lineIndex.append(int(vertex_index) - 1)
                        if i + 1 < len(vertex_indices):
                            lineLength += np.linalg.norm(np.array(networkVertex[int(vertex_indices[i + 1]) - 1]) - np.array(networkVertex[int(vertex_indices[i]) - 1]))
                    if lineLength > lineLengthMax:
                        lineLengthMax = lineLength
                        lineLengthMaxCount = line_count
                    networkLineIndex.append(lineIndex)
                    line_count += 1
                    # print(lineLength)
        # print(lineLengthMax)

        networkVertexArray = np.array(networkVertex)
        lineIndexMax = networkLineIndex[lineLengthMaxCount]
        headPoint = networkVertexArray[lineIndexMax[0]]
        tailPoint = networkVertexArray[lineIndexMax[-1]]
        rootNode = TreeNode(lineIndexMax)
        sameHeadPointIndices = np.where(np.linalg.norm(networkVertexArray - headPoint, axis=1) < 1e-7)[0]
        sameTailPointIndices = np.where(np.linalg.norm(networkVertexArray - tailPoint, axis=1) < 1e-7)[0]
        if len(sameHeadPointIndices) < len(sameTailPointIndices):  # 和下一级有连接的一端为尾端
            pass
        else:
            rootNode.val = list(reversed(rootNode.val))

        def buildTree(Node):  # 建立树结构
            sameTailPointIndices = np.where(np.linalg.norm(networkVertexArray - networkVertexArray[Node.val[-1]], axis=1) < 1e-7)[0]  # 寻找和尾端连接的所有顶点
            if len(sameTailPointIndices) == 1:
                return Node
            childrenHeadPointIndices = sameTailPointIndices[sameTailPointIndices != Node.val[-1]]  # 去除尾端index，保留下一级头部的index
            for lineIndex in networkLineIndex:
                if lineIndex[0] in childrenHeadPointIndices:
                    Node.children.append(buildTree(TreeNode(val=lineIndex, father=Node)))
                elif lineIndex[-1] in childrenHeadPointIndices:
                    Node.children.append(buildTree(TreeNode(val=list(reversed(lineIndex)), father=Node)))  # 保证边的顺序为头部在前尾端在后
                else:
                    continue
            return Node
        tree = buildTree(rootNode)

        networkVertexArray_original = networkVertexArray.copy()
        self.networkVertexArray = np.dot(self.R_model, networkVertexArray_original.T).T * 0.01 + self.t_model
        def upSampleNetwork(root): # 对Network的顶点进行上采样，并且保存
            if root is None: 
                return
            else:
                vertexList = []
                for i in range(len(root.val) - 1):
                    index1 = root.val[i]
                    index2 = root.val[i + 1]
                    vertex1 = self.networkVertexArray[index1]
                    vertex2 = self.networkVertexArray[index2]
                    interval = 0.01  # decimetre
                    if np.linalg.norm(vertex2 - vertex1) < interval:  # 两点之间距离已经足够小
                        vertexList.append(vertex1)
                    else:
                        point_num = int(np.linalg.norm(vertex2 - vertex1) / interval)
                        real_interval_3d = (vertex2 - vertex1) / point_num
                        for k in range(point_num):
                            vertex = vertex1 + k * real_interval_3d
                            vertexList.append(vertex)
                vertexList.append(self.networkVertexArray[root.val[-1]])  # 加上尾端顶点
                root.vertexArray = np.array(vertexList)
                for child in root.children:
                    upSampleNetwork(child)
                return
        upSampleNetwork(rootNode)

        # self.total_list = []
        # def maxDepth(root, cur_id):  # 验证树结构的正确性
        #     if root is None: 
        #         return
        #     else:
        #         self.total_list += root.val
        #         for index, child in enumerate(root.children):
        #             child.id = '{}-{}'.format(cur_id + 1, index)
        #             maxDepth(child, cur_id + 1)
        #         return
        # rootNode.id = '0-0'
        # maxDepth(rootNode, 0)
        # self.total_list.sort()
        # # print(self.total_list)

        def levelOrder(root):  # 层序遍历，并且为每个节点添加ID
            res = []  # 结果
            res_node = []
            level_index = 0
            if root :  
                queue = [root]  # 第一层
            else:
                return res
            while len(queue):  # 当下一层没有子节点后停止遍历
                n = len(queue)
                r = []
                r_node = []
                for index in range(n):
                    node = queue.pop(0)  # 弹出第一个值
                    node.id = '{}-{}'.format(level_index, index)
                    r.append(node.val)
                    r_node.append(node)
                    for child in node.children:
                        if child:
                            queue.append(child)
                    # if node.left:  # 左子树判断
                    #     queue.append(node.left)
                    # if node.right:  # 右子树判断
                    #     queue.append(node.right)
                level_index += 1
                res.append(r)  # 加入一层的结果
                res_node.append(r_node)
            return res, res_node
        self.levelIndices, self.levelNodesList = levelOrder(rootNode)
        self.levelNodesListAug = []
        for i, levelNodes in enumerate(self.levelNodesList):
            if i == 0:
                self.levelNodesListAug.append(levelNodes)
                continue
            node_list = levelNodes
            for node in self.levelNodesList[i - 1]:
                if len(node.children) == 0:  # if this node is tail node, pass to next level
                    node_list.append(node)
            self.levelNodesListAug.append(node_list)

        # 找出所有尾节点（没有任何头节点与之重合）
        tail_index_list = []
        head_index_list = []
        for levelIndicesList in self.levelIndices:
            for IndicesList in levelIndicesList:
                head_index_list.append(IndicesList[0])
                tail_index_list.append(IndicesList[-1])
        for head_index in head_index_list:
            for tail_index in tail_index_list:
                if np.linalg.norm(self.networkVertexArray[head_index] - self.networkVertexArray[tail_index]) < 1e-7:
                    tail_index_list.remove(tail_index)
                    break
        tail_vertex_array = []
        for tail_index in tail_index_list:
            tail_vertex_array.append(self.networkVertexArray[tail_index])
        self.tail_vertex_array = np.array(tail_vertex_array)

        nearest_tail_index = np.linalg.norm(self.networkVertexArray - self.centerlineArray[0], axis=1).argmin()  # 找中心线尾端对应的叶子节点
        self.targetLeafNode = None
        def findTargetNode(root):
            if root is None: 
                return
            else:
                for child in root.children:
                    if nearest_tail_index in child.val:
                        self.targetLeafNode = child
                    findTargetNode(child)
                return
        findTargetNode(rootNode)

        tmp_node = self.targetLeafNode  # 找中心线对应的节点
        centerline_node_list = []
        while tmp_node:
            centerline_node_list.append(tmp_node)
            tmp_node = tmp_node.father
        self.centerline_node_list = list(reversed(centerline_node_list))
        self.centerline_lineVertexArray_list = [node.vertexArray for node in self.centerline_node_list]
        self.centerline_networkVertexArray = np.concatenate(self.centerline_lineVertexArray_list, axis=0)
        self.centerline_networkVertex_list = [node.val for node in self.centerline_node_list]
        self.rootNode = rootNode
        self.already_passed_node_list = []
        self.global_node_count = 0

        for lineIndices in self.centerline_networkVertex_list:
            p.addUserDebugLine(self.networkVertexArray[lineIndices[0]], self.networkVertexArray[lineIndices[-1]], lineColorRGB=(0, 0, 1), lifeTime=0, lineWidth=3)
        
        p.addUserDebugLine(self.networkVertexArray[self.centerline_networkVertex_list[0][0]], self.networkVertexArray[self.centerline_networkVertex_list[0][-1]], lineColorRGB=(0, 0, 1), lifeTime=0, lineWidth=3)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.history_length = args.history_length


    def generate_heatmap_target(self, heatmap_size, landmarks):
        if landmarks[0] > heatmap_size[0] or landmarks[0] < 0 or \
            landmarks[1] > heatmap_size[1] or landmarks[1] < 0:
            return np.zeros(heatmap_size, dtype=np.uint8)
        mu = 0; sigma=20
        x, y = np.meshgrid(np.arange(heatmap_size[0]),
                        np.arange(heatmap_size[1]))
        dst = np.sqrt((x - landmarks[0])**2+(y - landmarks[1])**2)
        # plt.imshow(dst)
        # plt.show()

        # lower normal part of gaussian
        normal = 1/(np.sqrt(2.0 * np.pi) * sigma)

        # Calculating Gaussian filter
        gauss = np.exp(-((dst-mu)**2 / (2.0 * sigma**2))) * normal
        gauss = (gauss - 0) / (np.max(gauss) - 0) * 255
        gauss = gauss.astype(np.uint8)

        return gauss


    def smooth_centerline(self, centerlineArray, win_width=10):
        centerlineArray_smoothed = np.zeros_like(centerlineArray)
        for i in range(len(centerlineArray)):
            left_bound = i - win_width
            right_bound = i + win_width
            if left_bound < 0: left_bound = 0
            if right_bound > len(centerlineArray): right_bound = len(centerlineArray)
            centerlineArray_smoothed[i] = np.mean(centerlineArray[left_bound : right_bound], axis=0)
        return centerlineArray_smoothed


    def random_start_point(self, rand_index=None):
        centerline_length = len(self.centerlineArray)
        if not rand_index:
            rand_index = np.random.choice(np.arange(int(2 * centerline_length / 3), centerline_length - 3), 1)[0]
        pos_vector = self.centerlineArray[rand_index - 2] - self.centerlineArray[rand_index + 2]
        pitch = np.arcsin(pos_vector[2] / np.linalg.norm(pos_vector))
        if pos_vector[0] > 0:
            yaw = -np.arccos(pos_vector[1] / np.sqrt(pos_vector[0] ** 2 + pos_vector[1] ** 2))  # 相机绕自身坐标系旋转，Y轴正前，X轴正右，Z轴正上，yaw绕Z轴，pitch绕X轴，先yaw后pitch
        else:
            yaw = np.arccos(pos_vector[1] / np.sqrt(pos_vector[0] ** 2 + pos_vector[1] ** 2))
        quat = p.getQuaternionFromEuler([pitch, 0, yaw])
        R = p.getMatrixFromQuaternion(quat)
        R = np.reshape(R, (3, 3))

        rand_start_point = self.centerlineArray[centerline_length - 1]
        inside_flag = 0
        distance = 5
        while inside_flag == 0 or distance < 0.1:
            rand_start_point_in_original_cor = np.array([(np.random.rand() - 0.5) * 20, 0, (np.random.rand() - 0.5) * 20]) / 100
            rand_start_point = np.dot(R, rand_start_point_in_original_cor) + self.centerlineArray[rand_index]

            # Collision detection (check whether a point is inside the object by vtk and use the closest vertex)
            transformed_point = np.dot(np.linalg.inv(self.R_model), rand_start_point - self.t_model) * 100
            # transformed_point_vtk_cor = np.array([-transformed_point[0], -transformed_point[1], transformed_point[2]]) # x and y here is opposite to those in the world coordinate system
            transformed_point_vtk_cor = np.array([transformed_point[0], transformed_point[1], transformed_point[2]]) # x and y here is opposite to those in the world coordinate system
            pointId_target = self.pointLocator.FindClosestPoint(transformed_point_vtk_cor)
            cloest_point_vtk_cor = np.array(self.vtkdata.GetPoint(pointId_target))
            distance = np.linalg.norm(transformed_point_vtk_cor - cloest_point_vtk_cor)
            points = vtk.vtkPoints()
            points.InsertNextPoint(transformed_point_vtk_cor)
            pdata_points = vtk.vtkPolyData()
            pdata_points.SetPoints(points)
            enclosed_points_filter = vtk.vtkSelectEnclosedPoints()
            enclosed_points_filter.SetInputData(pdata_points)
            enclosed_points_filter.SetSurfaceData(self.vtkdata)
            enclosed_points_filter.SetTolerance(0.000001)  # should not be too large
            enclosed_points_filter.Update()
            inside_flag = int(enclosed_points_filter.GetOutput().GetPointData().GetArray('SelectedPoints').GetTuple(0)[0])
        
        rand_pitch = (np.random.rand() - 0.5) * 140
        rand_yaw = (np.random.rand() - 0.5) * 140
        return rand_pitch, rand_yaw, rand_start_point[0], rand_start_point[1], rand_start_point[2]


    def indexFromDistance(self, centerlineArray, count, distance):
        centerline_size = len(centerlineArray)
        start_index = count
        cur_index = start_index
        centerline_length = 0
        if cur_index <= 0:
            return False
        while(1):
            length_diff = np.linalg.norm(centerlineArray[cur_index - 1] - centerlineArray[cur_index])
            centerline_length += length_diff
            cur_index -= 1
            if cur_index <= 0:
                return False
            if centerline_length > distance:
                return cur_index
    
    def indexFromDistanceReversed(self, centerlineArray, count, distance):
        centerline_size = len(centerlineArray)
        start_index = count
        cur_index = start_index
        centerline_length = 0
        if cur_index >= centerline_size - 1:
            return False
        while(1):
            length_diff = np.linalg.norm(centerlineArray[cur_index + 1] - centerlineArray[cur_index])
            centerline_length += length_diff
            cur_index += 1
            if cur_index >= centerline_size - 1:
                return False
            if centerline_length > distance:
                return cur_index
            

    def rodriguez_rotation_matrix(self, vector1, vector2):
        # Normalize the input vectors
        v1 = vector1 / np.linalg.norm(vector1)
        v2 = vector2 / np.linalg.norm(vector2)

        # Compute the cross product of the normalized vectors
        cross_product = np.cross(v1, v2)

        # Compute the dot product of the normalized vectors
        dot_product = np.dot(v1, v2)

        # Compute the skew-symmetric cross product matrix
        skew_matrix = np.array([[0, -cross_product[2], cross_product[1]],
                                [cross_product[2], 0, -cross_product[0]],
                                [-cross_product[1], cross_product[0], 0]])

        # Compute the rotation matrix
        rotation_matrix = np.eye(3) + skew_matrix + np.dot(skew_matrix, skew_matrix) * (1 / (1 + dot_product))

        return rotation_matrix
    

    def nms(self, center_points, confidence_score, threshold):
        # If no bounding boxes, return empty list
        if len(center_points) == 0:
            return [], []

        # Bounding boxes
        center = np.array(center_points)

        # coordinates of bounding boxes
        center_x = center[:, 0]
        center_y = center[:, 1]

        # Confidence scores of bounding boxes
        score = np.array(confidence_score)

        # Picked bounding boxes
        picked_score = []
        picked_centers = []

        # Compute areas of bounding boxes
        # areas = (end_x - start_x + 1) * (end_y - start_y + 1)
        masks = []
        for index in range(len(center_points)):
            mask = np.zeros((200, 200))
            cv2.circle(mask, (center_x[index], center_y[index]), radius=40, color=(255, 255, 255), thickness=-1)
            # # mask = np.mean(mask, axis=-1)
            # cv2.imshow("mask", mask)
            # cv2.waitKey(1)
            mask = (mask / 255).astype(np.bool)
            masks.append(mask)
        masks = np.array(masks)

        # Sort by confidence score of bounding boxes
        order = np.argsort(score)

        # Iterate bounding boxes
        while order.size > 0:
            # The index of largest confidence score
            index = order[-1]

            # Pick the bounding box with largest confidence score
            # picked_boxes.append(bounding_boxes[index])
            picked_centers.append(center_points[index])
            picked_score.append(confidence_score[index])

            intersection = np.logical_and(masks[index], masks[order[:-1]])
            union = np.logical_or(masks[index], masks[order[:-1]])

            # Compute the ratio between intersection and union
            # ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)
            ratio = np.sum(intersection, axis=(1, 2)) / np.sum(union, axis=(1, 2))

            left = np.where(ratio < threshold)
            order = order[left]

        return picked_centers, picked_score

   

    def run(self, net, target_detection_net, epoch=None, net_transfer=None, transform_func=None, transform_func_transfer=None, training=True):

        if training:
            saving_root = os.path.join("train_set", "centerlines_with_dagger", self.centerline_name + "-dagger" + str(epoch))
            if not os.path.exists(saving_root):
                os.mkdir(saving_root)
            actions_saving_dir = os.path.join(saving_root, "actions.txt")
            images_saving_root = os.path.join(saving_root, "rgb_images")
            ref_images_saving_root = os.path.join(saving_root, "ref_rgb_images")
            condition_saving_root = os.path.join(saving_root, "condition_images")
            targets_saving_root = os.path.join(saving_root, "targets_images")
            depth_saving_root = os.path.join(saving_root, "depth_images")
            pred_depth_saving_root = os.path.join(saving_root, "pred_depth_images")
            pred_targets_saving_root = os.path.join(saving_root, "pred_targets_images")
            if not os.path.exists(images_saving_root):
                os.mkdir(images_saving_root)
            if not os.path.exists(ref_images_saving_root):
                os.mkdir(ref_images_saving_root)
            if not os.path.exists(condition_saving_root):
                os.mkdir(condition_saving_root)
            if not os.path.exists(targets_saving_root):
                os.mkdir(targets_saving_root)
            if not os.path.exists(depth_saving_root):
                os.mkdir(depth_saving_root)
            if not os.path.exists(pred_depth_saving_root):
                os.mkdir(pred_depth_saving_root)
            if not os.path.exists(pred_targets_saving_root):
                os.mkdir(pred_targets_saving_root)
            f = open(actions_saving_dir, 'w')

        current_time = datetime.datetime.now()

        for level_id, levelIndicesList in enumerate(self.levelIndices):
            for IndicesList in levelIndicesList:
                middle_vertex = self.networkVertexArray[IndicesList[int(len(IndicesList) / 2)]]

                # Collision detection (check whether a point is inside the object by vtk and use the closest vertex)
                transformed_point = np.dot(np.linalg.inv(self.R_model), middle_vertex - self.t_model) * 100
                # transformed_point_vtk_cor = np.array([-transformed_point[0], -transformed_point[1], transformed_point[2]]) # x and y here is opposite to those in the world coordinate system
                transformed_point_vtk_cor = np.array([transformed_point[0], transformed_point[1], transformed_point[2]]) # x and y here is opposite to those in the world coordinate system
                pointId_target = self.pointLocator.FindClosestPoint(transformed_point_vtk_cor)
                cloest_point_vtk_cor = np.array(self.vtkdata.GetPoint(pointId_target))
                distance = np.linalg.norm(transformed_point_vtk_cor - cloest_point_vtk_cor)
                points = vtk.vtkPoints()
                points.InsertNextPoint(transformed_point_vtk_cor)
                pdata_points = vtk.vtkPolyData()
                pdata_points.SetPoints(points)
                enclosed_points_filter = vtk.vtkSelectEnclosedPoints()
                enclosed_points_filter.SetInputData(pdata_points)
                enclosed_points_filter.SetSurfaceData(self.vtkdata)
                enclosed_points_filter.SetTolerance(0.000001)  # should not be too large
                enclosed_points_filter.Update()
                inside_flag = int(enclosed_points_filter.GetOutput().GetPointData().GetArray('SelectedPoints').GetTuple(0)[0])

        count = len(self.centerlineArray) - 1

        if training:
            # pitch, yaw, x, y, z = self.random_start_point()
            start_index = len(self.centerlineArray) - 3
            pitch, yaw, x, y, z = self.random_start_point(rand_index=start_index)
            yaw = 0
            pitch = 0
        else:
            start_index = len(self.centerlineArray) - 3
            pitch, yaw, x, y, z = self.random_start_point(rand_index=start_index)
            yaw = 0
            # pitch = -89.9999
            pitch = 0
            # x = self.centerlineArray[len(self.centerlineArray) - 1, 0]
            # y = self.centerlineArray[len(self.centerlineArray) - 1, 1]
            # z = self.centerlineArray[len(self.centerlineArray) - 1, 2]

        quat_init = p.getQuaternionFromEuler([pitch, 0, yaw])
        # quat_init = np.array([35, 78, 33, 80]) / np.linalg.norm(np.array([35, 78, 33, 80]))
        R = p.getMatrixFromQuaternion(quat_init)
        R = np.reshape(R, (3, 3))
        quat = dcm2quat(R)
        t = np.array([x, y, z])
        t_current = t
        # t_origin = self.centerlineArray[-1]
        t_origin = self.centerline_node_list[0].vertexArray[0]  # 图的起点作为原点
        pos_vector = self.centerlineArray[count - 1] - self.centerlineArray[count]
        pos_vector_old = pos_vector
        pos_vector_passed = pos_vector
        pos_vector_base = [0, 1, 0]
        R_base = np.identity(3)
        R_current = self.rodriguez_rotation_matrix(pos_vector_base, pos_vector)

        quat = p.getQuaternionFromEuler([np.pi / 2, 0, 0])
        R_fix = p.getMatrixFromQuaternion(quat)
        R_fix = np.reshape(R_fix, (3, 3))
        R_test = np.dot(R_fix, R)
        delta_R = np.identity(3)
        # R_current = np.dot(R_test, np.linalg.inv(R_fix))
        # t_current = t

        min_nearest_centerline_point_dist = np.inf
        min_level = 0
        for i in range(len(self.centerline_node_list)):
            nearest_centerline_point_dist = np.linalg.norm(self.centerline_node_list[i].vertexArray - t, axis=1).min()
            nearest_network_centerline_point_sim_cor_index = np.linalg.norm(self.centerline_node_list[i].vertexArray - t, axis=1).argmin()
            if nearest_centerline_point_dist < min_nearest_centerline_point_dist:
                min_nearest_centerline_point_dist = nearest_centerline_point_dist
                min_level = i

        for i in range(len(self.centerlineArray) - 1):
            p.addUserDebugLine(self.centerlineArray[i], self.centerlineArray[i + 1], lineColorRGB=[0, 1, 0], lifeTime=0, lineWidth=3)
        
        path_length = 0
        path_centerline_error_list = []
        path_centerline_length_list = []
        path_centerline_ratio_list = []
        safe_distance_list = []
        path_centerline_pred_position_list = []

        level = 0
        level_old = level
        command_buffer = []
        error_stop_detection_buffer = []
        rgb_img_list = []
        state_buffer = deque([], maxlen=self.history_length)
        # frame_buffer = torch.zeros(2, 84, 84, 3, device=self.device)
        count_step = 0
        MAXLEVEL = len(self.centerline_node_list) - 1
        backward_flag = 0
        explore_state = 0  # node: 0; query: 1; forward: 2; backword: 3
        covered_leaf_node_list = []
        full_passed_trajectory = []
        reach_tail_flag = False
        target_explore_node = None

        # Show network and current postion
        def create_binary_tree(node, graph, parent=None):
            if node is None:
                return

            graph.add_node(node.id)

            if parent is not None:
                graph.add_edge(parent, node.id)

            for child in node.children:
                create_binary_tree(child, graph, parent=node.id)

        # Create a networkx graph
        plt.figure(figsize=(10, 5))
        graph = nx.DiGraph()
        create_binary_tree(self.rootNode, graph)

        # Draw the binary tree using matplotlib
        pos = graphviz_layout(graph, prog='dot')
        node_color_list = []
        for node in graph.nodes():
            if node in self.already_passed_node_list:
                node_color_list.append('orange')
            else:
                node_color_list.append('lightgrey')
        nx.draw_networkx(graph, pos, with_labels=True, node_size=200, \
                        node_color=node_color_list, font_size=5, arrows=True)
        plt.axis('off')
        plt.title('Bronchial Tree')
        # plt.tight_layout()
        plt.show()

        count_env = 0
        N_period = np.random.randint(2, 6)
        inside_flag = 1
    
        while 1:

            #------------Randomly rotate model------------#
            # t_origin_in_sim_cor = np.dot(np.linalg.inv(self.R_model), t_origin - self.t_model) * 100
            # t_in_sim_cor = np.dot(np.linalg.inv(self.R_model), t - self.t_model) * 100
            # self.rand_roll = (np.random.rand() - 0.5) * 2 * np.pi
            # euler = p.getEulerFromQuaternion([np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2])
            # self.quaternion_model = p.getQuaternionFromEuler([np.pi / 2, 0, 0])
            # self.matrix_model = p.getMatrixFromQuaternion(self.quaternion_model)
            # self.R_model = np.reshape(self.matrix_model, (3, 3))
            # self.t_model = np.array([0, 0, 5])
            # count_env += 1
            # t_origin = np.dot(self.R_model, t_origin_in_sim_cor) * 0.01 + self.t_model
            # t = np.dot(self.R_model, t_in_sim_cor) * 0.01 + self.t_model
            # # p.resetBasePositionAndOrientation(self.airwayBodyId, [0, 0, 5], self.quaternion_model)
            delta_t_model = 0
            count_env += 1

            print("level:", level)
            
            tic = time.time()


            if np.linalg.norm(t_current - t) > 1e-7:
                full_passed_trajectory.append(t_current)
            t_current = t

            # Locate the ground truth node
            nearest_tail_index = np.linalg.norm(self.networkVertexArray - t_current, axis=1).argmin()  # 找当前位置对应的节点
            self.targetLeafNode = None
            def findTargetNode(root):
                if root is None:
                    return
                else:
                    if nearest_tail_index in root.val:
                        self.targetLeafNode = root
                        return
                    else:
                        for child in root.children:
                            findTargetNode(child)
            findTargetNode(self.rootNode)
            nearest_index_in_target_node = np.linalg.norm(self.targetLeafNode.vertexArray - t_current, axis=1).argmin()  # 找当前位置所处目标节点的下标
            # nearest_tail_index = np.linalg.norm(self.networkVertexArray - t_current, axis=1).argmin()  # 找当前位置所处目标节点的下标

            R_current = np.dot(self.rodriguez_rotation_matrix(pos_vector_old, pos_vector), R_current)
            pos_vector_old = pos_vector

            command_exp = [0, 0, 0, 0, 1, 0]

            # Get image
            # pitch = pitch / 180 * np.pi + np.pi / 2
            # yaw = yaw / 180 * np.pi
            # quat = p.getQuaternionFromEuler([pitch, 0, yaw])
            # R = p.getMatrixFromQuaternion(quat)
            # R = np.reshape(R, (3, 3))
            quat = p.getQuaternionFromEuler([np.pi / 2, 0, 0])
            R_fix = p.getMatrixFromQuaternion(quat)
            R_fix = np.reshape(R_fix, (3, 3))
            R = np.dot(R_current, R_fix)
            # R_test = np.dot(R_test, delta_R)
            rot = Rotation.from_matrix(R)
            angle = rot.as_euler('xyz')
            # print("angle:", angle / np.pi * 180)
            pose = np.identity(4)
            pose[:3, 3] = t
            pose[:3, :3] = R
            light_intensity = 0.3
            self.scene.clear()
            # self.fuze_node = Node(mesh=self.fuze_mesh, scale=self.meshScale, rotation=self.quaternion_model, translation=self.t_model + np.array([0, 0.1 * np.sin(count_env / 5), 0]))
            self.fuze_node = Node(mesh=self.fuze_mesh, scale=self.meshScale, rotation=self.quaternion_model, translation=self.t_model + delta_t_model)
            self.scene.add_node(self.fuze_node)
            spot_l = SpotLight(color=np.ones(3), intensity=light_intensity,
                innerConeAngle=0, outerConeAngle=np.pi/2, range=1)
            spot_l_node = self.scene.add(spot_l, pose=pose)
            cam_node = self.scene.add(self.cam, pose=pose)
            self.scene.set_pose(spot_l_node, pose)
            self.scene.set_pose(cam_node, pose)
            rgb_img, depth_img = self.r.render(self.scene)
            rgb_img = rgb_img[:, :, :3]

            mean_intensity = np.mean(rgb_img)
            count_AE = 0
            min_light_intensity = 0.001
            max_light_intensity = 20
            while np.abs(mean_intensity - 140) > 20:
                if count_AE > 1000:
                    break
                if np.abs(min_light_intensity - light_intensity) < 1e-5 or np.abs(max_light_intensity - light_intensity) < 1e-5:
                    break
                if mean_intensity > 140:
                    max_light_intensity = light_intensity
                    light_intensity = (min_light_intensity + max_light_intensity) / 2
                else:
                    min_light_intensity = light_intensity
                    light_intensity = (min_light_intensity + max_light_intensity) / 2
                self.scene.clear()
                self.scene.add_node(self.fuze_node)
                spot_l = SpotLight(color=np.ones(3), intensity=light_intensity,
                        innerConeAngle=0, outerConeAngle=np.pi/2, range=1)
                spot_l_node = self.scene.add(spot_l, pose=pose)
                cam_node = self.scene.add(self.cam, pose=pose)
                self.scene.set_pose(spot_l_node, pose)
                self.scene.set_pose(cam_node, pose)
                rgb_img, depth_img = self.r.render(self.scene)
                rgb_img = rgb_img[:, :, :3]
                mean_intensity = np.mean(rgb_img)
                count_AE += 1

            rgb_img = cv2.resize(rgb_img, (200, 200))
            rgb_img = np.transpose(rgb_img, axes=(2, 0, 1))
            rgb_img_show = np.transpose(rgb_img, axes=(1, 2, 0))[:, :, ::-1] # RGB to BGR for show
            # plt.imshow(self.rgb_img)
            # plt.show()
            depth_img[depth_img == 0] = 0.5  # 去除空洞
            intrinsic_matrix = np.array([[175 / 1.008, 0, 200],
                                        [0, 175 / 1.008, 200],
                                        [0, 0, 1]])
            rgb_img_show = cv2.resize(rgb_img_show, (400, 400))
            depth_img_copy = cv2.resize(depth_img, (400, 400))
            T_current = np.identity(4)
            T_current[:3, :3] = R_current
            T_current[:3, 3] = t_current - t_origin

            if explore_state == -1:  # 获取新的参考图但不建立新节点
                ref_rgb_img = rgb_img.copy()
                if transform_func:
                    ref_rgb_img_PIL = Image.fromarray(np.transpose(ref_rgb_img, axes=(1, 2, 0)))
                    ref_rgb_img_tensor = transform_func(ref_rgb_img_PIL).unsqueeze(0)
                    ref_rgb_img_tensor = ref_rgb_img_tensor.to(device=self.device, dtype=torch.float32)
                else:
                    ref_rgb_img_tensor = torch.tensor(ref_rgb_img.copy()).unsqueeze(0)
                    ref_rgb_img_tensor = ref_rgb_img_tensor.to(device=self.device, dtype=torch.float32)
                explore_state = 2

            # if (level == 0 and count == len(self.centerlineArray) - 1) and (not backward_flag):
            if explore_state == 0 and (level == 0 and count == len(self.centerlineArray) - 1):
                # condition_img_single = self.generate_heatmap_target((200, 200), point1 / 2)
                ref_rgb_img = rgb_img.copy()
                if transform_func:
                    ref_rgb_img_PIL = Image.fromarray(np.transpose(ref_rgb_img, axes=(1, 2, 0)))
                    ref_rgb_img_tensor = transform_func(ref_rgb_img_PIL).unsqueeze(0)
                    ref_rgb_img_tensor = ref_rgb_img_tensor.to(device=self.device, dtype=torch.float32)
                else:
                    ref_rgb_img_tensor = torch.tensor(ref_rgb_img.copy()).unsqueeze(0)
                    ref_rgb_img_tensor = ref_rgb_img_tensor.to(device=self.device, dtype=torch.float32)
                predicted_targets, _ = target_detection_net(ref_rgb_img_tensor)
                targets_img = predicted_targets.data.cpu().numpy()

                # targets_img = np.zeros((200, 200), dtype=np.uint8)
                # if self.targetLeafNode:
                #     node = self.targetLeafNode
                #     lineVertex_in_current_cor = np.dot(np.linalg.inv(T_current), np.concatenate([node.vertexArray - t_origin, np.ones((len(node.vertexArray), 1))], axis=1).T).T[:, :3]
                #     lineVertex_in_camera_cor = np.array([lineVertex_in_current_cor[:, 0], -lineVertex_in_current_cor[:, 2], lineVertex_in_current_cor[:, 1]]).T
                #     lineVertex_in_image_cor = np.dot(intrinsic_matrix, lineVertex_in_camera_cor.T).T / np.abs(np.expand_dims(lineVertex_in_camera_cor[:, 2], axis=1))
                #     point = lineVertex_in_image_cor[-1]
                #     for i in range(len(lineVertex_in_image_cor)):
                #         point = lineVertex_in_image_cor[len(lineVertex_in_image_cor) - 1 - i]
                #         if int(point[1] + 0.5) < 400 and int(point[0] + 0.5) < 400 and \
                #             int(point[1] + 0.5) >= 0 and int(point[0] + 0.5) >= 0:
                #             depth_value1 = depth_img[int(point[1] + 0.5)][int(point[0] + 0.5)]
                #             if (depth_value1 > lineVertex_in_camera_cor[len(lineVertex_in_image_cor) - 1 - i][2] and lineVertex_in_camera_cor[len(lineVertex_in_image_cor) - 1 - i][2] > 0):
                #                 cv2.circle(rgb_img_show, (int((point[0] - 200) + 200 + 0.5), int((point[1] - 200) + 200 + 0.5)), 3, (255, 0, 0), -1)
                #                 break
                #     heatmap = self.generate_heatmap_target((200, 200), point / 2)
                #     targets_img = np.max(np.array([targets_img, heatmap]), axis=0, keepdims=False).astype(np.uint8)
                # targets_img = np.expand_dims(targets_img, axis=(0, 1))

                order = np.argsort(targets_img.ravel())
                scores = targets_img.ravel()[order[-2000:]]
                positions_x = order[-2000:] % targets_img.shape[3]
                positions_y = (order[-2000:] / targets_img.shape[3]).astype(np.int)
                center_points = [k for k in zip(positions_x, positions_y)]
                picked_centers, picked_score = self.nms(center_points, scores, 0.3)
                for (center_x, center_y), confidence in zip(picked_centers, picked_score):
                    cv2.circle(rgb_img_show, (center_x * 2, center_y * 2), 20, color=(255, 0, 0), thickness=2)
                condition_list = []
                for center_point in picked_centers:
                    condition_list.append(self.generate_heatmap_target((200, 200), center_point))
                frontiers = list(range(len(condition_list)))
                root_explore_node = ExploreTreeNode(ref_img=ref_rgb_img, condition_list=condition_list, frontiers=frontiers, id='0-0')
                # root_explore_node = ExploreTreeNode(ref_img=ref_rgb_img, condition_list=condition_list, frontiers=frontiers, id=self.targetLeafNode.id)
                cur_explore_node = root_explore_node
                # cur_passed_trajectory = []
                # cur_explore_node.passed_trajectory = cur_passed_trajectory

                # Show network and current postion
                def create_binary_tree(node, graph, parent=None):
                    if node is None:
                        return

                    graph.add_node(node.id)

                    if parent is not None:
                        graph.add_edge(parent, node.id)

                    for child in node.children:
                        create_binary_tree(child, graph, parent=node.id)

                # Create a networkx graph
                graph = nx.DiGraph()
                create_binary_tree(root_explore_node, graph)

                # Draw the binary tree using matplotlib
                pos = graphviz_layout(graph, prog='dot')
                plt.ion()
                plt.figure(figsize=(10, 5))

                plt.clf()
                node_color_list = []
                for node in graph.nodes():
                    if node in self.already_passed_node_list:
                        node_color_list.append('orange')
                    else:
                        node_color_list.append('lightgrey')
                nx.draw_networkx(graph, pos, with_labels=True, node_size=200, \
                                node_color=node_color_list, font_size=5, arrows=True)
                plt.axis('off')
                plt.title('Bronchial Tree')
                # plt.tight_layout()
                # plt.savefig("BinaryTree.png")
                plt.pause(0.001)
                plt.ioff()

                explore_state = 1  # query
            
            # if  (level != level_old) and (not backward_flag):
            if explore_state == 0:
                # condition_img_single = self.generate_heatmap_target((200, 200), point1 / 2)
                ref_rgb_img = rgb_img.copy()
                if transform_func:
                    ref_rgb_img_PIL = Image.fromarray(np.transpose(ref_rgb_img, axes=(1, 2, 0)))
                    ref_rgb_img_tensor = transform_func(ref_rgb_img_PIL).unsqueeze(0)
                    ref_rgb_img_tensor = ref_rgb_img_tensor.to(device=self.device, dtype=torch.float32)
                else:
                    ref_rgb_img_tensor = torch.tensor(ref_rgb_img.copy()).unsqueeze(0)
                    ref_rgb_img_tensor = ref_rgb_img_tensor.to(device=self.device, dtype=torch.float32)
                predicted_targets, _ = target_detection_net(ref_rgb_img_tensor)
                targets_img = predicted_targets.data.cpu().numpy()

                order = np.argsort(targets_img.ravel())
                scores = targets_img.ravel()[order[-2000:]]
                positions_x = order[-2000:] % targets_img.shape[3]
                positions_y = (order[-2000:] / targets_img.shape[3]).astype(np.int)
                center_points = [k for k in zip(positions_x, positions_y)]
                picked_centers, picked_score = self.nms(center_points, scores, 0.3)
                for (center_x, center_y), confidence in zip(picked_centers, picked_score):
                    cv2.circle(rgb_img_show, (center_x * 2, center_y * 2), 20, color=(255, 0, 0), thickness=2)
                condition_list = []
                picked_centers = list(sorted(picked_centers, key=lambda y: y[0]))
                for center_point in picked_centers:
                    condition_list.append(self.generate_heatmap_target((200, 200), center_point))
                frontiers = list(range(len(condition_list)))
                self.global_node_count += 1
                new_explore_node = ExploreTreeNode(ref_img=ref_rgb_img, condition_list=condition_list, frontiers=frontiers, father=cur_explore_node, id='{}-{}'.format(level, self.global_node_count))
                # new_explore_node = ExploreTreeNode(ref_img=ref_rgb_img, condition_list=condition_list, frontiers=frontiers, father=cur_explore_node, id=self.targetLeafNode.id)
                cur_explore_node.children.append(new_explore_node)
                cur_explore_node = new_explore_node
                cur_explore_node.passed_trajectory = cur_passed_trajectory  # save the passed trajectory, means how to reach current node
                cur_explore_node.passed_pose_trajectory = cur_passed_pose_trajectory

                # Show network and current postion
                def create_binary_tree(node, graph, parent=None):
                    if node is None:
                        return

                    graph.add_node(node.id)

                    if parent is not None:
                        graph.add_edge(parent, node.id)

                    for child in node.children:
                        create_binary_tree(child, graph, parent=node.id)

                # Create a networkx graph
                graph = nx.DiGraph()
                create_binary_tree(root_explore_node, graph)

                # Draw the binary tree using matplotlib
                pos = graphviz_layout(graph, prog='dot')
                plt.close()
                plt.ion()
                plt.figure(figsize=(10, 5))

                plt.clf()
                node_color_list = []
                for node in graph.nodes():
                    if node in self.already_passed_node_list:
                        node_color_list.append('orange')
                    else:
                        node_color_list.append('lightgrey')
                nx.draw_networkx(graph, pos, with_labels=True, node_size=200, \
                                node_color=node_color_list, font_size=5, arrows=True)
                plt.axis('off')
                plt.title('Bronchial Tree')
                # plt.tight_layout()
                # plt.savefig("BinaryTree.png")
                plt.pause(0.001)
                plt.ioff()

                # if level > 2:
                # if len(self.targetLeafNode.children) == 0 and len(self.targetLeafNode.vertexArray) - nearest_index_in_target_node < 10:
                if np.linalg.norm(self.tail_vertex_array - t_current, axis=1).min() < 0.05 or inside_flag == 0:
                    reach_tail_flag = True
                    # level += 1
                    # if not (self.targetLeafNode in covered_leaf_node_list):
                    #     covered_leaf_node_list.append(self.targetLeafNode)
                    tmp_explore_node = cur_explore_node
                    target_explore_node = None
                    while tmp_explore_node.father:
                        level -= 1
                        if len(tmp_explore_node.father.frontiers) > 0:
                            target_explore_node = tmp_explore_node.father
                            break
                        tmp_explore_node = tmp_explore_node.father
                    if target_explore_node is None:
                        print(covered_leaf_node_list)
                        print(len(covered_leaf_node_list))
                        break
                    backward_trajectory = []
                    backward_pose_trajectory = []
                    tmp_explore_node = cur_explore_node
                    while tmp_explore_node != target_explore_node:
                        backward_trajectory.append(list(reversed(tmp_explore_node.passed_trajectory)))
                        backward_pose_trajectory += list(reversed(tmp_explore_node.passed_pose_trajectory))
                        tmp_explore_node = tmp_explore_node.father
                    backward_trajectory = np.concatenate(backward_trajectory).tolist()
                    # backward_trajectory = list(reversed(backward_trajectory))
                    # condition_img_single = target_explore_node.condition_list[target_explore_node.frontiers[0]]
                    condition_img_single = np.zeros((200, 200), dtype=np.uint8)
                    ref_rgb_img = target_explore_node.ref_img
                    condition_img = np.stack([ref_rgb_img[0, :, :], ref_rgb_img[1, :, :], condition_img_single], axis=0)
                    explore_state = 3  # backward
                else:
                    explore_state = 1  # query

            if explore_state == 1:
                ref_rgb_img = cur_explore_node.ref_img
                chosen_frontier = cur_explore_node.frontiers.pop(0)
                condition_img_single = cur_explore_node.condition_list[chosen_frontier]
                condition_img = np.stack([ref_rgb_img[0, :, :], ref_rgb_img[1, :, :], condition_img_single], axis=0)
                cur_passed_trajectory = []
                cur_passed_pose_trajectory = []

                ref_rgb_img_grey = cv2.cvtColor(np.transpose(ref_rgb_img, axes=(1, 2, 0)), cv2.COLOR_RGB2GRAY)
                if transform_func:
                    ref_rgb_img_PIL = Image.fromarray(ref_rgb_img_grey)
                    ref_rgb_img_tensor = transform_func(ref_rgb_img_PIL)
                    ref_rgb_img_tensor = ref_rgb_img_tensor.to(device=self.device, dtype=torch.float32)
                rgb_img_list_all = []
                rgb_img_list_all.append(ref_rgb_img_tensor)

                explore_state = 2  # forward

            if explore_state == 2:
                # rgb_img_small = cv2.resize(rgb_img.transpose(1, 2, 0), (84, 84), interpolation=cv2.INTER_LINEAR)
                # ref_rgb_img_small = cv2.resize(ref_rgb_img.transpose(1, 2, 0), (84, 84), interpolation=cv2.INTER_LINEAR)
                # condition_img_single_small = cv2.resize(condition_img_single, (84, 84), interpolation=cv2.INTER_LINEAR)
                # state = np.stack([cv2.cvtColor(rgb_img_small, cv2.COLOR_BGR2GRAY), cv2.cvtColor(ref_rgb_img_small, cv2.COLOR_BGR2GRAY), condition_img_single_small], axis=-1)
                # state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).div_(255).unsqueeze(0)
                    
                # command = [0, 0, 0, 0, 0, 0]
                # command[4] = predicted_stop[0].argmax().cpu().item()
                # command[5] = predicted_stop[0].argmin().cpu().item()

                rgb_img_grey = cv2.cvtColor(np.transpose(rgb_img, axes=(1, 2, 0)), cv2.COLOR_RGB2GRAY)
                ref_rgb_img_grey = cv2.cvtColor(np.transpose(ref_rgb_img, axes=(1, 2, 0)), cv2.COLOR_RGB2GRAY)
                # High-level network inference
                if transform_func:
                    # rgb_img_PIL = Image.fromarray(np.transpose(rgb_img, axes=(1, 2, 0)))
                    # ref_rgb_img_PIL = Image.fromarray(np.transpose(ref_rgb_img, axes=(1, 2, 0)))
                    # condition_img_PIL = Image.fromarray(np.transpose(condition_img, axes=(1, 2, 0)))
                    rgb_img_PIL = Image.fromarray(rgb_img_grey)
                    ref_rgb_img_PIL = Image.fromarray(ref_rgb_img_grey)
                    condition_img_PIL = Image.fromarray(condition_img_single)
                    rgb_img_tensor = transform_func(rgb_img_PIL)
                    ref_rgb_img_tensor = transform_func(ref_rgb_img_PIL)
                    condition_img_tensor = transform_func(condition_img_PIL)
                    rgb_img_tensor = rgb_img_tensor.to(device=self.device, dtype=torch.float32)
                    ref_rgb_img_tensor = ref_rgb_img_tensor.to(device=self.device, dtype=torch.float32)
                    condition_img_tensor = condition_img_tensor.to(device=self.device, dtype=torch.float32)
                else:
                    raise NotImplementedError()
                rgb_img_list_all.append(rgb_img_tensor)
                
                # if len(rgb_img_list) == 0:
                #     for _ in range(self.history_length):
                #         rgb_img_list.append(rgb_img_tensor)
                # else:
                #     rgb_img_list.pop(0)
                #     rgb_img_list.append(rgb_img_tensor)

                rgb_img_list = []
                for i in range(self.history_length):
                    rgb_img_list.append(rgb_img_list_all[int(i / (self.history_length - 1) * (len(rgb_img_list_all) - 1) + 0.5)])

                rgb_img_tensor = torch.cat(rgb_img_list, dim=0)
                ref_rgb_img_tensor = ref_rgb_img_tensor.repeat(self.history_length, 1, 1)
                condition_img_tensor = condition_img_tensor.repeat(self.history_length, 1, 1)
                input_tensor = torch.stack([rgb_img_tensor, ref_rgb_img_tensor, condition_img_tensor], dim=-1).unsqueeze(0)

                predicted_action, predicted_stop = net(input_tensor)
                print("soft max:", torch.nn.functional.softmax(predicted_stop))
                command = [0, 0, 0, 0, 0, 0]
                if torch.nn.functional.softmax(predicted_stop)[0][1] > 0.5:
                    command[4] = 0
                    command[5] = 1
                else:
                    command[4] = 1
                    command[5] = 0
                # command[4] = predicted_stop[0].argmin().cpu().item()
                # command[5] = predicted_stop[0].argmax().cpu().item()

                # Randomly choose expert or novice policy for training
                if training:
                    beta = 0.5
                else:
                    beta = 0
                expert_prob = beta
                if np.random.rand() < expert_prob:
                    command = command_exp
                    cv2.putText(rgb_img_show, 'EXP', (300, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                else:
                    cv2.putText(rgb_img_show, 'NOV', (300, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

                if command == [1, 0, 0, 0, 0, 0]:
                    cv2.putText(rgb_img_show, 'Up', (20, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                elif command == [0, 1, 0, 0, 0, 0]:
                    cv2.putText(rgb_img_show, 'Left', (20, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                elif command == [0, 0, 1, 0, 0, 0]:
                    cv2.putText(rgb_img_show, 'Down', (20, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                elif command == [0, 0, 0, 1, 0, 0]:
                    cv2.putText(rgb_img_show, 'Right', (20, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                elif command == [0, 0, 0, 0, 1, 0]:
                    cv2.putText(rgb_img_show, 'Straight', (20, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                elif command == [0, 0, 0, 0, 0, 1]:
                    cv2.putText(rgb_img_show, 'Stop', (20, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                else:
                    raise NotImplementedError()
                if command_exp == [1, 0, 0, 0, 0, 0]:
                    cv2.putText(rgb_img_show, 'Up', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                elif command_exp == [0, 1, 0, 0, 0, 0]:
                    cv2.putText(rgb_img_show, 'Left', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                elif command_exp == [0, 0, 1, 0, 0, 0]:
                    cv2.putText(rgb_img_show, 'Down', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                elif command_exp == [0, 0, 0, 1, 0, 0]:
                    cv2.putText(rgb_img_show, 'Right', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                elif command_exp == [0, 0, 0, 0, 1, 0]:
                    cv2.putText(rgb_img_show, 'Straight', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                elif command_exp == [0, 0, 0, 0, 0, 1]:
                    cv2.putText(rgb_img_show, 'Stop', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                else:
                    raise NotImplementedError()
                
                # cur_explore_node.passed_trajectory.append(t)
                cur_passed_trajectory.append(t)
                cur_passed_pose_trajectory.append(pos_vector)

                # Collision detection (check whether a point is inside the object by vtk and use the closest vertex)
                transformed_point = np.dot(np.linalg.inv(self.R_model), t - (self.t_model + delta_t_model)) * 100
                # transformed_point_vtk_cor = np.array([-transformed_point[0], -transformed_point[1], transformed_point[2]]) # x and y here is opposite to those in the world coordinate system
                transformed_point_vtk_cor = np.array([transformed_point[0], transformed_point[1], transformed_point[2]]) # x and y here is opposite to those in the world coordinate system
                pointId_target = self.pointLocator.FindClosestPoint(transformed_point_vtk_cor)
                cloest_point_vtk_cor = np.array(self.vtkdata.GetPoint(pointId_target))
                distance = np.linalg.norm(transformed_point_vtk_cor - cloest_point_vtk_cor)
                points = vtk.vtkPoints()
                points.InsertNextPoint(transformed_point_vtk_cor)
                pdata_points = vtk.vtkPolyData()
                pdata_points.SetPoints(points)
                enclosed_points_filter = vtk.vtkSelectEnclosedPoints()
                enclosed_points_filter.SetInputData(pdata_points)
                enclosed_points_filter.SetSurfaceData(self.vtkdata)
                enclosed_points_filter.SetTolerance(0.000001)  # should not be too large
                enclosed_points_filter.Update()
                inside_flag = int(enclosed_points_filter.GetOutput().GetPointData().GetArray('SelectedPoints').GetTuple(0)[0])
                
                # Level change
                if len(command_buffer) < 10:
                    command_buffer.append(command)
                else:
                    command_buffer.pop(0)
                    command_buffer.append(command)
                if command == [0, 0, 0, 0, 0, 1] or np.linalg.norm(self.tail_vertex_array - t_current, axis=1).min() < 0.05 or inside_flag == 0:
                    # if command == [0, 0, 0, 0, 0, 1]:
                    #     rgb_img_list_all.pop(-1)  # don't add redundant stop frame
                    if (np.sum(np.array(command_buffer)[:, :5]) == 0 and len(command_buffer) == 10) or np.linalg.norm(self.tail_vertex_array - t_current, axis=1).min() < 0.05 or inside_flag == 0:
                        # if target_explore_node and np.linalg.norm(target_explore_node.passed_trajectory[-1] - t_current) < 0.05 and np.linalg.norm(self.tail_vertex_array - t_current, axis=1).min() > 0.1:  # 如果这次前进是先退后再前进，且停止位置距离分岔口节点很近，那么不建立新的子节点
                        #     # explore_state = -1
                        #     pass
                        if level != 0 and np.linalg.norm(cur_explore_node.passed_trajectory[-1] - t_current) < 0.02 and np.linalg.norm(self.tail_vertex_array - t_current, axis=1).min() > 0.1:  # 如果和当前节点离得很近，那么不建立新的子节点
                            explore_state = -1
                            # pass
                        else:
                            level += 1
                            command_buffer = []
                            error_stop_detection_buffer = []
                            explore_state = 0  # Node state
                        # level += 1
                        # command_buffer = []
                        # error_stop_detection_buffer = []
                        # explore_state = 0  # Node state

            if explore_state == 3:
                if len(backward_trajectory) == 0:

                    cur_explore_node = target_explore_node  # back to node that has frontiers
                    rgb_img_grey = cv2.cvtColor(np.transpose(cur_explore_node.ref_img, axes=(1, 2, 0)), cv2.COLOR_RGB2GRAY)
                    ref_rgb_img_grey = cv2.cvtColor(np.transpose(ref_rgb_img, axes=(1, 2, 0)), cv2.COLOR_RGB2GRAY)
                    # High-level network inference
                    if transform_func:
                        # rgb_img_PIL = Image.fromarray(np.transpose(rgb_img, axes=(1, 2, 0)))
                        # ref_rgb_img_PIL = Image.fromarray(np.transpose(ref_rgb_img, axes=(1, 2, 0)))
                        # condition_img_PIL = Image.fromarray(np.transpose(condition_img, axes=(1, 2, 0)))
                        rgb_img_PIL = Image.fromarray(rgb_img_grey)
                        ref_rgb_img_PIL = Image.fromarray(ref_rgb_img_grey)
                        condition_img_PIL = Image.fromarray(condition_img_single)
                        rgb_img_tensor = transform_func(rgb_img_PIL)
                        ref_rgb_img_tensor = transform_func(ref_rgb_img_PIL)
                        condition_img_tensor = transform_func(condition_img_PIL)
                        rgb_img_tensor = rgb_img_tensor.to(device=self.device, dtype=torch.float32)
                        ref_rgb_img_tensor = ref_rgb_img_tensor.to(device=self.device, dtype=torch.float32)
                        condition_img_tensor = condition_img_tensor.to(device=self.device, dtype=torch.float32)
                    else:
                        raise NotImplementedError()
                    
                    if len(rgb_img_list) == 0:
                        for _ in range(10):
                            rgb_img_list.append(rgb_img_tensor)
                    else:
                        rgb_img_list.pop(0)
                        rgb_img_list.append(rgb_img_tensor)
                    
                    rgb_img_tensor = torch.cat(rgb_img_list, dim=0)
                    ref_rgb_img_tensor = ref_rgb_img_tensor.repeat(self.history_length, 1, 1)
                    condition_img_tensor = condition_img_tensor.repeat(self.history_length, 1, 1)
                    input_tensor = torch.stack([rgb_img_tensor, ref_rgb_img_tensor, condition_img_tensor], dim=-1).unsqueeze(0)

                    predicted_action, predicted_stop = net(input_tensor)
                    command = [0, 0, 0, 0, 0, 0]
                    if torch.nn.functional.softmax(predicted_stop)[0][1] > 0.5:
                        command[4] = 0
                        command[5] = 1
                    else:
                        command[4] = 1
                        command[5] = 0
                    # command[4] = predicted_stop[0].argmin().cpu().item()
                    # command[5] = predicted_stop[0].argmax().cpu().item()
                    
                    explore_state = 1  # query

            yaw_in_camera_cor = predicted_action.squeeze(0).cpu().data.numpy()[0] * (np.pi / 2)  # pred action belongs to (-1, 1) and needs scaling to (-pi / 2, pi / 2)
            pitch_in_camera_cor = predicted_action.squeeze(0).cpu().data.numpy()[1] * (np.pi / 2)  # pred action belongs to (-1, 1) and needs scaling to (-pi / 2, pi / 2)
            quat_in_camera_cor = p.getQuaternionFromEuler([pitch_in_camera_cor, yaw_in_camera_cor, 0])
            R_in_camera_cor = p.getMatrixFromQuaternion(quat_in_camera_cor)
            R_in_camera_cor = np.reshape(R_in_camera_cor, (3, 3))
            # pose_in_camera_cor = np.dot(R_in_camera_cor, [0, 0, 1 / 400])
            pose_in_camera_cor = np.dot(R_in_camera_cor, [0, 0, 1 / 200])
            # pose_in_camera_cor = np.dot(R_in_camera_cor, [0, 0, 1 / 100])
            pose_in_current_cor = np.array([pose_in_camera_cor[0], pose_in_camera_cor[2], -pose_in_camera_cor[1]])
            pos_vector = np.dot(R_current, pose_in_current_cor)
            
            pos_vector_norm = np.linalg.norm(pos_vector)
            if pos_vector_norm < 1e-5:
                raise NotImplementedError()
            
            if explore_state == 2:
                if command == [0, 0, 0, 0, 0, 1]:
                    t = t
                    pos_vector = pos_vector_passed
                    # continue
                else:
                    t = t + pos_vector
                    pos_vector_passed = pos_vector
            if explore_state == 3:
                t = np.array(backward_trajectory.pop(0))
                pos_vector = np.array(backward_pose_trajectory.pop(0))
                pos_vector_passed = pos_vector
                print("t:", t)

            pose_cur_in_current_cor = np.dot(np.linalg.inv(R_current), pos_vector)
            pose_cur_in_camera_cor = np.array([pose_cur_in_current_cor[0], -pose_cur_in_current_cor[2], pose_cur_in_current_cor[1]])
            pose_cur_in_image_cor = np.dot(intrinsic_matrix, pose_cur_in_camera_cor) / np.abs(pose_cur_in_camera_cor[2])
            cv2.arrowedLine(rgb_img_show, (200, 200), (int((pose_cur_in_image_cor[0] - 200) + 200 + 0.5), int((pose_cur_in_image_cor[1] - 200) + 200 + 0.5)), (0, 0, 255), thickness=2, line_type=8, shift=0, tipLength=0.3)

            cv2.imshow("TAR", cv2.resize(targets_img.squeeze(), (200, 200)))
            cv2.imshow("COND", cv2.resize(np.transpose(condition_img, axes=(1, 2, 0)), (200, 200)))
            # cv2.waitKey(1)
            cv2.imshow("REF RGB IMAGE", cv2.resize(np.transpose(ref_rgb_img, axes=(1, 2, 0))[:, :, ::-1], (200, 200)))
            # cv2.waitKey(1)
            cv2.imshow("RGB IMAGE", rgb_img_show)
            # cv2.imshow("RGB IMAGE", np.transpose(rgb_img, axes=(1, 2, 0))[:, :, ::-1])
            cv2.moveWindow("TAR", 1000, 50)
            cv2.moveWindow("COND", 750, 50)
            cv2.moveWindow("REF RGB IMAGE", 1250, 50)
            cv2.moveWindow("RGB IMAGE", 200, 400)
            cv2.waitKey(1)

            # # Veering outside the lane
            # nearest_centerline_point_sim_cor_index = np.linalg.norm(self.centerlineArray - t, axis=1).argmin()
            # lane_width = nearest_centerline_point_sim_cor_index / (len(self.centerlineArray) - 1) * 0.08 + 0.02  # minimal width is 2mm, maximal width is 10mm 
            # nearest_distance_to_centerline = np.linalg.norm(self.centerlineArray - t, axis=1).min()
            # if nearest_distance_to_centerline > lane_width:
            #     break
            
            # # When backward, ensure robot not outside the bronchus. Use normal to adjust robot's position.
            # if distance < 1:
            #     normal = self.normals.GetTuple(pointId_target)
            #     if backward_flag:
            #         t = t - pos_vector_back
            #         normal_array = -np.array(normal)
            #         normal_array = np.dot(self.R_model, normal_array)
            #         t = t + normal_array / np.linalg.norm(normal_array) * 1 / 200

            # # Error stop detection
            # if len(error_stop_detection_buffer) < 20:
            #     error_stop_detection_buffer.append(command)
            # else:
            #     error_stop_detection_buffer.pop(0)
            #     error_stop_detection_buffer.append(command)
            #     if np.sum(np.array(error_stop_detection_buffer)[:, :5]) == 0:
            #         break
            
            # print("point1, -1", point1, lineVertex_in_image_cor[-1])
                
            count -= 1
            count_step += 1

        # 分级计算覆盖率
        full_passed_trajectory = np.array(full_passed_trajectory)
        for level_id, levelIndicesList in enumerate(self.levelIndices):
            count = 0
            for IndicesList in levelIndicesList:
                tail_index = IndicesList[-1]
                if np.linalg.norm(full_passed_trajectory - self.networkVertexArray[tail_index], axis=1).min() < 0.1:
                    count += 1
            print("Level {}, coverage ratio {} / {} = {}".format(level_id, count, len(levelIndicesList), count / len(levelIndicesList)))

        fig = mlab.figure(bgcolor=(1,1,1))
        src = mlab.pipeline.add_dataset(self.vtkdata, figure=fig)
        surf = mlab.pipeline.surface(src, opacity=0.2, color=(206 / 255, 108 / 255, 131 / 255))
        networkVertexArray_original = np.dot(np.linalg.inv(self.R_model), (self.networkVertexArray - self.t_model).T).T * 100
        full_passed_trajectory_original = np.dot(np.linalg.inv(self.R_model), (full_passed_trajectory - self.t_model).T).T * 100
        # full_passed_trajectory_original = full_passed_trajectory_original[:200]
        for level_id, levelIndicesList in enumerate(self.levelIndices):
            for IndicesList in levelIndicesList:
                level_vertex_array = networkVertexArray_original[IndicesList]
                # level_vertex_array_reverse = level_vertex_array[::-1]
                # level_vertex_array = np.concatenate([level_vertex_array, level_vertex_array_reverse])
                mlab.plot3d([p[0] for p in level_vertex_array], [p[1] for p in level_vertex_array], [p[2] for p in level_vertex_array], color=(0, 1, 0), tube_radius=0.5, tube_sides=10, figure=fig)
        mlab.plot3d([p[0] for p in full_passed_trajectory_original], [p[1] for p in full_passed_trajectory_original], [p[2] for p in full_passed_trajectory_original], color=(1, 0, 0), tube_radius=1, tube_sides=10, figure=fig)
        mlab.view(azimuth=-90, elevation=90, distance=600, figure=fig)
        mlab.show()
        # mlab.savefig("results/{}-{}-{}-{}-{}-{}-explore_front.png".format(current_time.year, current_time.month, current_time.day, current_time.hour, current_time.minute, current_time.second), figure=fig, magnification=5)
        mlab.view(azimuth=0, elevation=90, distance=600, figure=fig)
        mlab.show()
        # mlab.savefig("results/{}-{}-{}-{}-{}-{}-explore_side.png".format(current_time.year, current_time.month, current_time.day, current_time.hour, current_time.minute, current_time.second), figure=fig, magnification=5)
        mlab.clf()
        mlab.close(fig)

        p.disconnect()
        self.r.delete()
        cv2.destroyAllWindows()
        plt.clf()
        plt.close()

        return