# Generated by Django 2.0.5 on 2018-08-05 10:03

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('display', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='information',
            name='image',
            field=models.ImageField(upload_to='uploads/'),
        ),
    ]
