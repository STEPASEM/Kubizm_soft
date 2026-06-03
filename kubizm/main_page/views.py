from django.shortcuts import render

def Index(request):
    return render(request, 'main_page/index.html')
