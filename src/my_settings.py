DATABASES = {
    'default' : {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'vidid', # db이름
        'USER': 'cdh', # 로그인-유저 명
        'PASSWORD': '0000',# 로그인- 비밀번호
        'HOST': 'localhost',
        'PORT': '3306',
    }
}
SECRET_KEY = '1234'