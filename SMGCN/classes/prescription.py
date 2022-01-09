#coding=utf-8
class Prescription(object):
    def __init__(self, symptoms, herbs):
        self.__symptoms = symptoms
        self.__herbs = herbs

    @property
    def symptoms(self):
        return self.__symptoms

    @property
    def herbs(self):
        return self.__herbs

    @symptoms.setter
    def symptoms(self, value):
        if isinstance(value, list):
            self.__symptoms = value
        else:
            print("error:不是整数列表！")

    @herbs.setter
    def herbs(self, value):
        if isinstance(value, list):
            self.__herbs = value
        else:
            print("error:不是整数列表！")












