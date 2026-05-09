import torch
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from torchvision.utils import save_image
from .load_bfm import load_BFM, load_attribute
from .BFM_obj import BFM
from .rendering import image_renderer, image_rendering


class Face09(BFM):
    def __init__(
        self,
        BFMmodel,
        shape_random,
        tex_random,
        is_attribute=False,
        angles=[0, 0, 0],
        translation=[0, 0, 0],
        scale=8e-04,
        shape_variations=5e02,
        **kwargs
    ):
        super().__init__(BFMmodel)
        self.shape_random = shape_random
        self.tex_random = tex_random
        self.is_attribute = is_attribute
        self.angles = angles
        self.translation = translation
        self.scale = scale
        self.shape_variations = shape_variations

        self.triangles = self.triangles

        if self.is_attribute == True:
            self.gender_coef = kwargs.get("gender_coef", None)
            self.age_shape_coef = kwargs.get("age_shape_coef", None)
            self.age_tex_coef = kwargs.get("age_tex_coef", None)
            self.height_coef = kwargs.get("height_coef", None)
            self.weight_coef = kwargs.get("weight_coef", None)

    def get_coefs(self):
        self.shape_coef = super().get_shape_coef(self.shape_random, False)[0]
        self.tex_coef = super().get_tex_coef(self.tex_random)

        return self.shape_coef, self.tex_coef

    def generate_maps(self, shape_coef, tex_coef, attribute_data=None):
        if self.is_attribute == True:
            self.shape_map = super().get_shape_map_09(
                None,
                1.0,
                True,
                attribute_data,
                self.gender_coef,
                self.age_shape_coef,
                self.height_coef,
                self.weight_coef,
            )
            tex_map = super().get_texture_map(None, self.age_tex_coef, attribute_data)
        elif self.is_attribute == False:
            self.shape_map = super().get_shape_map_09(
                shape_coef, self.shape_variations, False
            )
            tex_map = super().get_texture_map(tex_coef)

        self.shape_transformed = super().transform_shape(
            self.shape_map, self.angles, self.translation, self.scale
        )

        tex_map = torch.clamp(tex_map.reshape(self.points_dims), 0, 1)
        self.tex_map = tex_map[np.newaxis, :, :]
        texturesV = TexturesVertex(self.tex_map)

        face_mesh = Meshes(
            verts=[self.shape_transformed.to(device)],
            faces=[self.triangles.to(device)],
            textures=texturesV.to(device),
        )
        self.face = face_mesh

        return face_mesh

    def one_random_face(
        self,
        shape_coef,
        tex_coef,
        img_path=None,
        img_size=256,
        light_direction=((0, 0, 10),),
    ):
        if shape_coef == None or tex_coef == None:
            shape_coef, tex_coef = self.get_coefs()
        face_09 = self.generate_maps(shape_coef, tex_coef)
        renderer = image_renderer(img_size=img_size, light_direction=light_direction)
        self.image = image_rendering(renderer, face_09)
        image_torch = self.image.permute(2, 0, 1)

        try:
            img_path = img_path + ".png"
            save_image(image_torch, img_path)
        except:
            pass

    # def random_face_generator(self, num_faces, img_folder, img_size = 256, light_direction = ((0, 0, 10), )):
    #     for i in range(num_faces):
    #         img_path = img_folder + str(i)
    #         face_09 = self.one_random_face(None, None, img_path, img_size, light_direction)


# 2009 Model with Gender Attributes
NUM_IMAGES = 20
Gender_range = np.linspace(-4, 5, NUM_IMAGES)  # * 1e03
# Age shape and age textures need to be aligned/have the same dimension.
Age_shape_range = np.linspace(-30, 100, NUM_IMAGES)  # * 1e03
Age_tex_range = np.linspace(-30, 100, NUM_IMAGES)
Height_range = np.linspace(-50, 100, NUM_IMAGES)  # * 1e03
Weight_range = np.linspace(-30, 120, NUM_IMAGES)  # * 1e03


def attribute_generator(**kwargs):
    img_folder = kwargs.get("img_folder", np.random.randint(1000))
    gender_range = kwargs.get("gender_range", np.zeros(NUM_IMAGES))
    age_shape_range = kwargs.get("age_shape_range", np.zeros(NUM_IMAGES))
    age_tex_range = kwargs.get("age_tex_range", np.zeros(NUM_IMAGES))
    height_range = kwargs.get("height_range", np.zeros(NUM_IMAGES))
    weight_range = kwargs.get("weight_range", np.zeros(NUM_IMAGES))
    angles = kwargs.get("angles", [0, 0, 0])
    model_path = kwargs.get("model_path", None)
    attribute_path = kwargs.get("attribute_path", None)

    BFM_09 = load_BFM(model_path)
    attribute_data = load_attribute(attribute_path)

    for count, (gender, age_shape, age_tex, height, weight) in enumerate(
        zip(gender_range, age_shape_range, age_tex_range, height_range, weight_range)
    ):
        face_09 = Face09(
            BFM_09,
            False,
            False,
            True,
            gender_coef=gender * 1e03,
            age_shape_coef=age_shape * 1e03,
            age_tex_coef=age_tex,
            height_coef=height * 1e03,
            weight_coef=weight * 1e03,
        )
        face_mesh = face_09.generate_maps(None, None, attribute_data)

        renderer = image_renderer()
        image = image_rendering(renderer, face_mesh)
        image_torch = image.permute(2, 0, 1)
        img_path = img_folder + str(count)

        try:
            img_path = img_path + ".png"
            save_image(image_torch, img_path)
        except:
            pass
