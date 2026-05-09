"""Basel Face Model(BFM) base module."""

from typing import Optional, Union

import torch
from torch.distributions.normal import Normal
from pytorch3d.transforms import euler_angles_to_matrix, Rotate, Translate, Scale


class BFM(object):
    """Parent BFM object. Can be modified to accomodate different versions of the model."""

    def __init__(
        self,
        model: dict,
        num_faces: int,
        scale: float = 0.8,
        translation: Union[list, tuple] = [0, 0, 0],
        seed: Optional[int] = None,
        device: Optional[str] = None,
    ):
        """
        Args:
            model(dict):
                A dictionary of model parameters returned by using load_BFM function in load_bfm.py.
            num_faces(int):
                Number of faces to be generated
            scale(float):
                Scaling factor that adjusts shape coordinates and subsequently influences visual size of the face on the rendered images.
            seed(:obj:`int`, optional):
                Fixed seed integer for generating random face latents.
            device(:obj:`str`, optional)

        """
        self.model = model
        self.num_faces = num_faces

        self.points_dims = self.model["points_dims"]
        self.shape_dims = self.model["shape_dims"]
        self.tex_dims = self.model["tex_dims"]
        self.triangles = self.model["Cells"]

        try:
            self.expr_dims = self.model["expr_dims"]
        except:
            pass

        self.scale = scale
        self.translation = translation
        self.seed = seed

        if device is None:
            if torch.cuda.is_available():
                device = [torch.device("cuda:0")]
            else:
                device = [torch.device("cpu")]

        self.device = device

    def get_random_coef(self, dims: int):
        """
        Args:
            dims(int):
                The dimensions for the coefficient vector, which is used for the Karhunan-Loeve expansion in generating shape or texture map.
                Each coefficient is generated from Normal(0,1).
        Returns:
            coef (numpy.ndarray):
                (N, D). Batch size N.
        """
        if self.seed is not None:
            coef = torch.normal(
                mean=0,
                std=1,
                size=(self.num_faces, dims),
                generator=torch.Generator(device=self.device[0]).manual_seed(self.seed),
                device=self.device[0],
            )
        else:
            coef = torch.normal(
                mean=0, std=1, size=(self.num_faces, dims), device=self.device[0]
            )
        return coef

    def get_shape_coef(self, is_shape_random: bool, is_expr_random: bool):
        shape_coef, expr_coef = None, None
        if is_shape_random is True:
            shape_coef = self.get_random_coef(self.shape_dims)

        if is_expr_random is True:
            expr_coef = self.get_random_coef(self.expr_dims)

        return shape_coef, expr_coef

    def get_tex_coef(self, is_tex_random: bool):
        tex_coef = None
        if is_tex_random is True:
            tex_coef = self.get_random_coef(self.tex_dims)

        return tex_coef

    def get_angle(self, is_angle_random: bool):
        """Get euler angles in radians for head orientations of each face.

        Args:
            is_angle_random (bool):
                If True, generates random angles from truncated normal distributions, in the order of pitch, yaw, and roll.
                Faces are frontal otherwise.

        Returns:
            angles (torch.Tensor): (N,3)

        Note:
            Means, standard deviations, and the bounds of the truncated normal distributions were set by heuristics / common human head orientation range.
        """
        if self.seed is not None:
            torch.manual_seed(self.seed)

        if is_angle_random == True:
            means = torch.tensor(
                [0.0, 0.0, 0.0], dtype=torch.float32, device=self.device[0]
            )
            stds = torch.tensor(
                [0.174533, 0.523599, 0.122173],
                dtype=torch.float32,
                device=self.device[0],
            )
            angles_min = torch.tensor(
                [-0.349066, -1.5708, -0.244346],
                dtype=torch.float32,
                device=self.device[0],
            )
            angles_max = -angles_min

            dist = Normal(loc=means, scale=stds)
            angles = dist.sample((self.num_faces,))
            angles = torch.max(torch.min(angles, angles_max), angles_min)
        else:
            angles = [[0, 0, 0]] * self.num_faces
            angles = torch.tensor(angles, dtype=torch.float32, device=self.device[0])

        return angles

    def get_lighting_coef(self):
        """Generates random ambient light intensity and direction vector of the light from truncated normal distributions.

        Note:
            Means, standard deviations, and the bounds of the truncated normal distributions were set by heuristics and observations of lighting condition in the VGGFace2 dataset.
        """
        if self.seed is not None:
            torch.manual_seed(self.seed)

        direction_means = torch.tensor(
            [2.0, 0.0, 10.0], dtype=torch.float32, device=self.device[0]
        )
        direction_stds = torch.tensor(
            [2.0, 0.5, 10.0], dtype=torch.float32, device=self.device[0]
        )
        direction_min = torch.tensor(
            [0.0, 0.0, 5.0], dtype=torch.float32, device=self.device[0]
        )
        direction_max = torch.tensor(
            [5.0, 3.0, 30.0], dtype=torch.float32, device=self.device[0]
        )
        dist = Normal(loc=direction_means, scale=direction_stds)
        light_direction = dist.sample((self.num_faces,))
        light_direction = torch.max(
            torch.min(light_direction, direction_max), direction_min
        )

        intensity_mean = torch.tensor([0.6], dtype=torch.float32, device=self.device[0])
        intensity_std = torch.tensor([0.2], dtype=torch.float32, device=self.device[0])
        intensity_min, intensity_max = 0.5, 0.8
        dist = Normal(loc=intensity_mean, scale=intensity_std)
        intensity = dist.sample((self.num_faces,))
        intensity = intensity.clip(min=intensity_min, max=intensity_max)
        intensity = intensity.squeeze().repeat_interleave(3).reshape(-1, 3)

        return light_direction, intensity

    def get_shape_attributes(
        self,
        attribute_data,
        gender_coef=None,
        age_shape_coef=None,
        height_coef=None,
        weight_coef=None,
    ):
        dims = self.shape_dims
        shape_coef = torch.zeros((dims, 1))

        gender, age_shape, height, weight = (
            attribute_data["gender_shape"],
            attribute_data["age_shape"],
            attribute_data["height_shape"],
            attribute_data["weight_shape"],
        )
        gender, age_shape, height, weight = (
            torch.from_numpy(gender),
            torch.from_numpy(age_shape),
            torch.from_numpy(height),
            torch.from_numpy(weight),
        )

        if gender_coef is not None:
            shape_coef += gender_coef * gender[:dims]
        if age_shape_coef is not None:
            shape_coef += age_shape_coef * age_shape[:dims]
        if height_coef is not None:
            shape_coef += height_coef * height[:dims]
        if weight_coef is not None:
            shape_coef += weight_coef * weight[:dims]

        return shape_coef

    def get_tex_attributes(self, attribute_data, age_tex_coef=None):
        dims = self.tex_dims
        tex_coef = torch.zeros((dims, 1))

        age_texture = torch.from_numpy(attribute_data["age_tex"])

        if age_tex_coef is not None:
            tex_coef += age_tex_coef * age_texture[:dims]

        return tex_coef

    def get_shape_map_19(
        self,
        shape_coef: Optional[torch.Tensor] = None,
        expr_coef: Optional[torch.Tensor] = None,
        angles: Optional[torch.Tensor] = None,
    ):
        """Generates random or average shape map with Basel Face Model 2019 parameters and shape/expression coefficients."""

        shape_mean = self.model["shapeMU"]  # (174609,)
        shape_deformPC = self.model["shapePC"]  # (174609, 199)
        shape_deformEV = self.model["shapeEV"]  # (199,)
        expr_mean = self.model["exprMU"]
        expr_deformPC = self.model["exprPC"]
        expr_deformEV = self.model["exprEV"]

        shape_params = shape_mean

        if shape_coef is None and expr_coef is None:
            shape_params = [shape_params.tolist()] * self.num_faces
            shape_params = torch.tensor(
                shape_params, device=self.device[0]
            )  # torch.tile

        else:
            if shape_coef is not None:
                shape_deformEV = shape_coef * shape_deformEV  # (EV_dims, num_faces)
                shape_params = shape_params + shape_deformEV.matmul(shape_deformPC)

            if expr_coef is not None:
                expr_deformEV = expr_coef * expr_deformEV
                shape_params = (
                    shape_params + expr_mean + expr_deformEV.matmul(expr_deformPC)
                )

        shape_map = shape_params.reshape(
            self.num_faces, self.points_dims[0], self.points_dims[1]
        )
        shape_map = self.transform_shape(
            shape_map, angles, scale=self.scale, translation=self.translation
        )

        return shape_map

    def get_texture_map(self, tex_coef: Optional[torch.Tensor] = None):
        """Generates random or average texture map with Basel Face Model 2019 parameters and texture coefficients."""

        tex_params = self.model["texMU"]
        tex_deformPC = self.model["texPC"]
        tex_deformEV = self.model["texEV"]

        if tex_coef is not None:
            tex_deformEV = tex_coef * tex_deformEV
            tex_params = tex_params + tex_deformEV.matmul(tex_deformPC)
        else:
            tex_params = [tex_params.tolist()] * self.num_faces
            tex_params = torch.tensor(tex_params, device=self.device[0])

        tex_map = tex_params.reshape(
            self.num_faces, self.points_dims[0], self.points_dims[1]
        )

        return tex_map

    def get_shape_map_09(
        self,
        shape_coef=None,
        shape_variations=1.0,
        is_attribute=False,
        attribute_data=None,
        gender_coef=None,
        age_shape_coef=None,
        height_coef=None,
        weight_coef=None,
    ):
        if shape_coef != None and is_attribute == True:
            return "Shape cannot be random if you're specifying attribute parameters."

        shape_params = self.model["shapeMU"]  # (142317, 1)
        shape_deformPC = self.model["shapePC"]  # (142317, 199)
        shape_deformEV = self.model["shapeEV"]  # (199, 1)

        if is_attribute == True:
            shape_coef = self.get_shape_attributes(
                attribute_data, gender_coef, age_shape_coef, height_coef, weight_coef
            )
            shape_deformEV = shape_coef * shape_deformEV * shape_variations
            shape_params = shape_params + shape_deformPC.matmul(shape_deformEV)
        elif shape_coef is not None:
            shape_deformEV = shape_coef * shape_deformEV * shape_variations
            shape_params = shape_params + shape_deformPC.matmul(shape_deformEV)

        shape_map = shape_params.reshape(self.points_dims)
        shape_transformed = self.transform_shape(
            shape_map, angles, scale=self.scale
        )  # scale=1e-03

        return shape_map

    def transform_shape(
        self,
        shape_map: torch.Tensor,
        angles: torch.Tensor,
        translation: Union[torch.Tensor, list] = [0, 0, 0],
        scale: float = 0.8,
    ):
        """Scale, rotate, and translate the shape map."""

        rotation = euler_angles_to_matrix(angles, convention="XYZ")
        rotate = Rotate(rotation, device=self.device[0])

        t1, t2, t3 = translation[0], translation[1], translation[2]
        translate = Translate(t1, t2, t3, device=self.device[0])

        scale = Scale(scale, dtype=torch.float32, device=self.device[0])
        transformed_shape = (
            rotate.compose(scale).compose(translate).transform_points(shape_map)
        )

        return transformed_shape
