from django.shortcuts import render

def Index(request):
    return render(request, 'main_page/index.html')

def o_we(request):
    return render(request, 'main_page/o_we.html')
