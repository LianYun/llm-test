import torch
from server import PromptServer

class Imageselector:
    CATEGORY = 'example'
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"images": ("IMAGE",),
                             "mode": (["brightest", "reddest", "greenest", "bluest"], )}}
    RETURN_TYPES = ("IMAGE", )
    FUNCATION = "choose_image"

    def choose_image(self, images, mode):
        batch_size = images.shape[0]
        brightness = list(torch.mean(image.flatten()).item() for image in images)
        if mode == "brightest":
            scores = brightness
        else:
            channel = 0 if mode=="reddest" else (1 if mode=="greenest" else 2)
            absolute = list(torch.mean(image[:, :, channel].flatten()).item() for image in images)
            scores = list( absolute[i]/(brightness[i]+1e-8) for i in range(batch_size) )
        best = scores.index(max(scores))
        result = images[best].unsqueeze(0)
        PromptServer.instance.send_sync("example.imageselector.textmessage", {"message": f"Picked image {best+1}"})
        return (result, )
    