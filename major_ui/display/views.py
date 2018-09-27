from django.shortcuts import render, get_object_or_404
from django.template import loader
from django.http import HttpResponse, HttpResponseRedirect

from .models import Information
from .forms import InformationForm
from major_project import sample_caption

# Create your views here.
def index(request):
    form = InformationForm(request.POST or None, request.FILES or None)
    if form.is_valid():
        instance = form.save(commit=False)
        instance.img_caption = sample_caption.get_caption(instance.image)
        instance.save()
        return HttpResponseRedirect(instance.get_image_caption_url())
    context = {
        "form":form,
    }
    return render(request, 'display/index.html', context)

def display_caption(request, img_id):
    instance = get_object_or_404(Information, pk = img_id)
    context = {
        'image':instance.image,
        'image_caption':instance.img_caption,
    }
    return render(request, 'display/image_caption.html', context)
