from django.db import models


class Artwork(models.Model):
    """
    Модель для хранения произведений искусства
    """
    STYLE_CHOICES = [
        ('analytical_cubism', 'Аналитический кубизм'),
        ('synthetic_cubism', 'Синтетический кубизм'),
        ('cubism', 'Кубизм'),
        ('not_cubism', 'Не кубизм'),
    ]

    style = models.CharField(
        max_length=50,
        choices=STYLE_CHOICES,
        verbose_name='Стиль'
    )
    title = models.CharField(max_length=200, verbose_name='Название')
    artist = models.CharField(max_length=200, verbose_name='Художник')
    year = models.CharField(max_length=20, verbose_name='Год')
    description = models.TextField(verbose_name='Описание')
    image_url = models.URLField(verbose_name='Ссылка на изображение')
    source_url = models.URLField(verbose_name='Ссылка на источник')
    order = models.IntegerField(default=0, verbose_name='Порядок')

    class Meta:
        ordering = ['style', 'order']
        verbose_name = 'Произведение'
        verbose_name_plural = 'Произведения'

    def __str__(self):
        return f"{self.title} - {self.artist}"
