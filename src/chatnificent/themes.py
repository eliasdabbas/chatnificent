from abc import ABC, abstractmethod


class Theme:
    @abstractmethod
    def cdn_link():
        return "https://bootstrap_link.com", "kskd8f2lalkf"  # <- integrity

    @abstractmethod
    def stylesheets():
        list_of_stylesheets = []  # and or filepaths
        return list_of_stylesheets

    @abstractmethod
    def js():
        list_of_scripts = []
        return list_of_scripts


class XYZBank(Theme):
    def cdn_link():
        pass

    def stylesheets():
        pass

    def js():
        pass
