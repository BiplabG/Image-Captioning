from django.db import models
from django.urls import reverse

# Create your models here.

image_path = 'uploads/'
class Information(models.Model):
    img_caption = models.CharField(max_length = 200)
    image = models.ImageField(upload_to = image_path)

    def get_image_caption_url(self):
        return reverse("image_with_caption", kwargs={"img_id":self.pk})