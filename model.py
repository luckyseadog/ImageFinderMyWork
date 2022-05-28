import abc
import torch
import clip
import numpy as np
from PIL import Image
import os
import glob
import pandas as pd
import math
from typing import List
from pathlib import Path


class DummyIndexer():
    def __init__(self):
        """
        Creates an empty index object
        """
        self.index = None

    def add(self, embs: np.ndarray):
        """
        Adds new embeddings embs in empty or existing index
        :param embs:
        :return:
        """
        if self.index is None:
            self.index = embs
        else:
            self.index = np.append(self.index, embs, axis=0)

    def train(self):
        """
        Not sure if this one is necessary here, left for compatibility with abstract class Indexer
        :return:
        """
        pass

    def find(self, query: np.ndarray, topn: int) -> (np.ndarray, np.ndarray):
        """
        Returns topn entries closest to the query vector
        :param query:
        :param topn:
        :return:
        """
        similarities = (self.index @ query.squeeze())
        best_photo_idx = (-similarities).argsort()
        D, I = similarities[best_photo_idx[:topn]], best_photo_idx[:topn]
        return D, I

    def save(self, file: str):
        """
        Saves data to npy file
        :param file:
        :return:
        """
        np.save(file, self.index)

    def load(self, file: str):
        """
        Loads data from npy file
        :param file:
        :return:
        """
        self.index = np.load(file)

class Embedder(abc.ABC):
    @abc.abstractmethod
    def encode_text(self, text):
        pass

    @abc.abstractmethod
    def encode_imgs(self, imgs):
        pass

class EmbedderCLIP(Embedder):
    def __init__(self, clip_model_name='RN101', device='cpu'):
        """
        :param clip_model_name:
        :param device:
        """
        self.device = device
        self.predictor, self.preprocess = clip.load(clip_model_name, device=device)

    def _tonumpy(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Detaches tensor from GPU and converts it to numpy array
        :return: numpy array
        """
        return tensor.cpu().detach().numpy()

    def encode_text(self, text: str) -> np.ndarray:
        """
        Returns text latent of the text input
        :param text:
        :return:
        """
        with torch.no_grad():
            # Encode it to a feature vector using CLIP
            text_latent = self.predictor.encode_text(clip.tokenize(text).to(self.device))
            text_latent /= text_latent.norm(dim=-1, keepdim=True)

        return self._tonumpy(text_latent)

    def encode_imgs(self, pil_imgs: List[Image.Image]) -> np.ndarray:
        """
        Returns image latents of a image batch
        :param pil_imgs: list of PIL images
        :return img_latents: numpy array of img latents
        """

        # Preprocess all photos
        photos_preprocessed = torch.stack([self.preprocess(photo) for photo in pil_imgs]).to(self.device)

        with torch.no_grad():
            # Encode the photos batch to compute the feature vectors and normalize them
            img_latents = self.predictor.encode_image(photos_preprocessed)
            img_latents /= img_latents.norm(dim=-1, keepdim=True)

        return self._tonumpy(img_latents)

class SearchModel():
    def __init__(self, embedder, indexer):
        self.embedder = embedder
        self.indexer = indexer
        self.images_dir = None
        self.imgs_path = None
        self.features_path = None

    def load_imgs(self, path: str):
        """
        Returns a list of names images in a given path
        :param path:
        :return:
        """
        self.images_dir = path
        photos_path = Path(self.images_dir)
        features_dir = str(photos_path.parents[0]) + '/features'
        # features_dir = general_features_dir + '/' + prefix
        self.features_path = Path(features_dir)
        self.imgs_path = list(photos_path.glob("*.png"))

        if not os.path.exists(features_dir):
            os.mkdir(features_dir)

        # if not os.path.exists(features_dir):
        #   os.mkdir(features_dir)

        if len(os.listdir(features_dir)) >= 2:
            self.imgs_path = list(pd.read_csv(f"{self.features_path}/photo_ids.csv")[
                                      'photo_id'])  # подумай, почему есть это условие: типа фото сохраняются в друге место?

    def load_img_urls(self):
        """
        In case we want to load imgs from a list of url
        :return:
        """
        pass

    def save_embs(self, batch_size=512) -> None:
        """
        Extracts image embeddings from embedder and adds them to indexer
        :param pil_imgs:
        :return:
        """

        if len(os.listdir(self.features_path)) >= 2:
            os.remove(str(self.features_path) + '/photo_ids.csv')
            os.remove(str(self.features_path) + '/features.npy')
            self.imgs_path = list(Path(self.images_dir).glob("*.png"))

        if not len(self.imgs_path) >= 512:
            batch_size = len(self.imgs_path)

        # Compute how many batches are needed
        print(batch_size)
        batches = math.ceil(len(self.imgs_path) / batch_size)

        # Process each batch
        for i in range(batches):
            print(f"Processing batch {i + 1}/{batches}")

            batch_ids_path = self.features_path / f"{i:010d}.csv"
            batch_features_path = self.features_path / f"{i:010d}.npy"

            # Only do the processing if the batch wasn't processed yet
            if not batch_features_path.exists():
                # Select the photos for the current batch
                batch_files = self.imgs_path[i * batch_size: min(len(self.imgs_path), (i + 1) * batch_size)]
                pil_batch = [Image.open(photo_file) for photo_file in batch_files]

                # Compute the features and save to a numpy file
                batch_features = self.embedder.encode_imgs(pil_batch)
                np.save(batch_features_path, batch_features)

                # Save the photo IDs to a CSV file
                photo_ids = [photo_file for photo_file in batch_files]
                photo_ids_data = pd.DataFrame(photo_ids, columns=['photo_id'])
                photo_ids_data.to_csv(batch_ids_path, index=False)

        # Load all numpy files
        features_list = [np.load(features_file) for features_file in sorted(self.features_path.glob("*.npy"))]

        # Concatenate the features and store in a merged file
        features = np.concatenate(features_list)
        np.save(self.features_path / "features.npy", features)

        # Load all the photo IDs
        photo_ids = pd.concat([pd.read_csv(ids_file) for ids_file in sorted(self.features_path.glob("*.csv"))])
        photo_ids.to_csv(self.features_path / "photo_ids.csv", index=False)

        for file in glob.glob('{}/0*.*'.format(self.features_path)):
            os.remove(file)

        self.indexer.load(str(self.features_path) + '/features.npy')

    def get_k_imgs(self, emb: np.ndarray, k: int):
        """
        Returns k indices of nearest image embeddings and respective distances for a given embedding emb
        :param emb:
        :param k:
        :return:
        """
        distances, indices = self.indexer.find(emb, k)
        return distances, np.array(self.imgs_path)[indices]
