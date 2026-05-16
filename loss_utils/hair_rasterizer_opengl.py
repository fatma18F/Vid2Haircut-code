import torch
import numpy as np
import moderngl as mgl
from typing import Tuple, Union

class PointCloudRasterizer:
    def __init__(
        self,
        n_points: int,
        resolution: Tuple[int, int] = (512, 512),
        point_size: float = 1.0
    ):
        self.resolution = resolution

        #with open('/fast/vsklyarova/Projects/SynthHair/data/shader.frag', 'r') as f, open('/fast/vsklyarova/Projects/SynthHair/data/shader.vert', 'r') as v:
        with open('shader.frag', 'r') as f, open('shader.vert', 'r') as v:
            self.frag = f.read()
            self.vert = v.read()

        self.ctx = mgl.create_context(standalone=True, backend='egl')
        self.ctx.point_size = point_size
        self.ctx.enable(mgl.DEPTH_TEST)
        self.ctx.depth_func = '<'

        self.prog = self.ctx.program(vertex_shader=self.vert, fragment_shader=self.frag)
        self.prog['P'].value = tuple(np.eye(4, dtype=np.float32).flatten())

        self.fbo = self.ctx.simple_framebuffer(resolution, components=1, dtype='f4')
        self.fbo.use()

        self.point_buf = self.ctx.buffer(reserve=n_points * (3 + 1) * 4)  # 3 for position, 1 for index
        self.point_vao = self.ctx.simple_vertex_array(
            self.prog,
            self.point_buf,
            'in_vert',
            'in_color'
        )

        self.left2right = torch.tensor([
            [-1.00,  0.00,  0.00,  0.00],
            [ 0.00,  1.00,  0.00,  0.00],
            [ 0.00,  0.00, -1.00,  0.00],
            [ 0.00,  0.00,  0.00,  1.00]
        ])

    @staticmethod
    def convert_intrinsic(
        k: torch.Tensor,
        image_size: Tuple[int, int],
        z_near: float = 0.01,
        z_far: float = 100.0
    ):
        w, h = image_size
        m = torch.eye(4)

        m[0][0] = 2.0 * k[0, 0] / w
        m[1][1] = 2.0 * k[1, 1] / h
        m[2][0] = 1.0 - 2.0 * k[0, 2] / w
        m[2][1] = 2.0 * k[1, 2] / h - 1.0
        m[2][2] = (z_near + z_far) / (z_near - z_far)
        m[2][3] = -2.0 * z_far * z_near / (z_near - z_far)
        m[3][2] = -1.0

        return m

    def rasterize(
        self,
        x: torch.Tensor,
        krt: Tuple[torch.Tensor, torch.Tensor],
        a: torch.Tensor,
        return_idx: bool
    ):
        verts = x.detach().reshape(-1, 3).cpu()

        K, RT = krt
        K = PointCloudRasterizer.convert_intrinsic(K, self.resolution, 0.01, 100)

        self.prog['P'].value = tuple((RT.T @ self.left2right @ K).cpu().numpy().astype(np.float32).flatten())

        self.point_buf.write(
            np.concatenate((verts, np.arange(len(verts)).reshape(-1, 1)), 1).astype(np.float32).flatten()
        )
        self.point_vao.render(mode=mgl.POINTS)

        r = np.frombuffer(self.fbo.read(components=1, dtype='f4'), dtype=np.float32)
        r = r.reshape(self.resolution + (1,))
        r = torch.tensor(r, dtype=torch.float32).flip(1).permute(2, 0, 1)

        self.fbo.clear()

        # Calculating indices
        r[r == 0] = -1
        i = torch.div(r, 1, rounding_mode='floor').type(torch.int32)[0]
        b = torch.zeros((a.shape[1],) + self.resolution)
        b[:, i >= 0] = a[i[i >= 0].long()].T
        if return_idx:
            return b, r
        else:
            return b

    def reset_geometry(self, n_points: int):
        if self.point_buf is not None:
            self.point_buf.release()
        self.point_buf = self.ctx.buffer(reserve=n_points * (3 + 1) * 4)
        if self.point_vao is not None:
            self.point_vao.release()
        self.point_vao = self.ctx.simple_vertex_array(self.prog, self.point_buf, 'in_vert', 'in_color')

        
        
        

class HairRasterizer:
    def __init__(
        self,
        n_hairs: int,
        strand_len: int,
        head_mesh: Union[Tuple[np.array, np.array], None] = None,
        resolution: Tuple[int, int] = (512, 512),
        line_width: float = 1.0
    ):

        self.hair_indices = HairRasterizer.get_strands_indices(n_hairs, strand_len)
        self.resolution = resolution

        #with open('/fast/vsklyarova/Projects/SynthHair/data/shader.frag', 'r') as f, open('/fast/vsklyarova/Projects/SynthHair/data/shader.vert', 'r') as v:
        with open('hader.frag', 'r') as f, open('/shader.vert', 'r') as v:
            self.frag = f.read()
            self.vert = v.read()

        self.ctx = mgl.create_context(standalone=True, backend='egl')
        self.ctx.line_width = line_width # Hack, move to geometry shader
        self.ctx.enable(mgl.DEPTH_TEST)
        self.ctx.depth_func = '<'
        self.strand_len = strand_len
        
        self.prog = self.ctx.program(vertex_shader=self.vert, fragment_shader=self.frag)

        self.prog['P'].value = tuple(np.eye(4, dtype=np.float32).flatten())
        #self.prog['RT'].value = tuple(np.eye(4, dtype=np.float32).flatten())

        self.fbo = self.ctx.simple_framebuffer(resolution, components=1, dtype='f4')
        self.fbo.use()

        self.hair_buf = self.ctx.buffer(reserve=n_hairs * strand_len * (3 + 1) * 4)

        self.head_buf = None
        self.head_vao = None

        if head_mesh is not None:
            self.reset_head_geometry(head_mesh)
            self.occlude_head = True
        else:
            self.occlude_head = False

        self.hair_vao = self.ctx.simple_vertex_array(
            self.prog,
            self.hair_buf,
            'in_vert',
            'in_color',
            index_buffer=self.ctx.buffer(self.hair_indices.flatten())
        )

        self.left2right = torch.tensor([
            [-1.00,  0.00,  0.00,  0.00],
            [ 0.00,  1.00,  0.00,  0.00],
            [ 0.00,  0.00, -1.00,  0.00],
            [ 0.00,  0.00,  0.00,  1.00]
        ])

#     @staticmethod
#     def convert_intrinsic(
#             k: torch.Tensor,
#             image_size: Tuple[int, int],
#             z_near: float = 0.01,
#             z_far: float = 100.0
#         ):

#         w, h = image_size
#         # Extract intrinsic parameters
#         f_x = k[0, 0]  # Focal length in x
#         f_y = k[1, 1]  # Focal length in y
#         c_x = k[0, 2]  # Principal point x
#         c_y = k[1, 2]  # Principal point y

#         # Initialize projection matrix
#         m = torch.zeros((4, 4), dtype=k.dtype, device=k.device)

#         # Scale factors for X and Y
#         m[0, 0] = 2.0 * f_x / w
#         m[1, 1] = 2.0 * f_y / h

#         # Principal point offsets
#         m[0, 2] = 2.0 * (c_x / w) - 1.0  # Offset for cx
#         m[1, 2] = 1.0 - 2.0 * (c_y / h)  # Offset for cy

#         # Depth mapping
#         m[2, 2] = -(z_far + z_near) / (z_far - z_near)
#         m[2, 3] = -2.0 * z_far * z_near / (z_far - z_near)

#         # Perspective division
#         m[3, 2] = -1.0

#         return m

    def convert_intrinsic(
        k: torch.tensor,
        image_size: Tuple[int,int],
        z_near: float = 0.01,
        z_far: float = 100
    ):

        w, h = image_size
        m = torch.eye(4)

        m[0][0] = 2.0 * k[0, 0] / w
        m[0][1] = 0.0
        m[0][2] = 0.0
        m[0][3] = 0.0

        m[1][0] = 0.0
        m[1][1] = 2 * k[1, 1] / h
        m[1][2] = 0.0
        m[1][3] = 0.0

        m[2][0] = 1.0 - 2.0 * k[0, 2] / w
        m[2][1] = 2.0 * k[1, 2] / h - 1.0
        m[2][2] = (z_near + z_far) / (z_near - z_far)
        m[2][3] = 1

        m[3][0] = 0.0
        m[3][1] = 0.0
        m[3][2] = 2.0 * z_far * z_near / (z_near - z_far)
        m[3][3] = 0.0

        return m

    @staticmethod
    def get_strands_indices(n_hairs: int, strand_len: int):
        indices = \
            np.arange(strand_len).repeat(2)[1: -1].reshape(1, strand_len * 2 - 2) + \
            np.arange(0, n_hairs * strand_len, strand_len).reshape(n_hairs, 1)

        return indices.astype(np.int32)

    def rasterize(
        self,
        x: torch.tensor,
        krt: Tuple[torch.tensor, torch.tensor],
        a: torch.tensor,
        return_idx: bool
    ):
        verts = x.detach().reshape(-1, 3).cpu()

        K, RT = krt
        K = HairRasterizer.convert_intrinsic(K, self.resolution, 0.01, 100)

        self.prog['P'].value = tuple((RT.T @ self.left2right @ K).cpu().numpy().astype(np.float32).flatten())

        self.hair_buf.write(
            np.concatenate((verts, np.arange(len(verts)).reshape(-1, 1)), 1).astype(np.float32).flatten()
        )
        self.hair_vao.render(mode=mgl.LINES)
        
        if self.occlude_head:
            self.head_vao.render(mode=mgl.TRIANGLES)

        r = np.frombuffer(self.fbo.read(components=1, dtype='f4'), dtype=np.float32)
        r = r.reshape(self.resolution + (1,))
        r = torch.tensor(r, dtype=torch.float32).flip(1).permute(2, 0, 1)

        self.fbo.clear()

        # Calculating indices
        r[r == 0] = -1
        i = torch.div(r,  self.strand_len, rounding_mode='floor').type(torch.int32)[0]
        b = torch.zeros((a.shape[1], ) + self.resolution)
        b[:, i >= 0] = a[i[i >= 0].long()].T
        if return_idx:
            return b, r
        else:
            return b

    def reset_hair_geometry(self, n_hairs,strand_len):
        self.strand_len = strand_len
        self.hair_indices = HairRasterizer.get_strands_indices(n_hairs, strand_len)
        if self.hair_buf is not None:
            self.hair_buf.release()
        self.hair_buf = self.ctx.buffer(reserve=n_hairs * strand_len * (3 + 1) * 4)
        if self.hair_vao is not None:
            self.hair_vao.release()
        self.hair_vao = self.ctx.simple_vertex_array(self.prog,self.hair_buf,'in_vert','in_color',index_buffer=self.ctx.buffer(self.hair_indices.flatten()))


    def reset_head_geometry(
        self,
        head_mesh: Union[Tuple[np.array, np.array], None]
    ):
        head_verts, head_faces = head_mesh

        if self.head_buf is not None:
            self.head_buf.release()

        self.head_buf = self.ctx.buffer(reserve=len(head_verts) * (3 + 1) * 4)

        self.head_buf.write(
            np.concatenate((head_verts, np.zeros((len(head_verts), 1))), 1).astype(np.float32).flatten()
        )

        if self.head_vao is not None:
            self.head_vao.release()

        self.head_vao = self.ctx.simple_vertex_array(
            self.prog,
            self.head_buf,
            'in_vert',
            'in_color',
            index_buffer=self.ctx.buffer(head_faces.astype(np.int32).flatten())
        )
