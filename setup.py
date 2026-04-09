from setuptools import setup
from setuptools.extension import Extension

try:
    from Cython.Distutils import build_ext
    use_cython = True
except ImportError:
    from setuptools.command.build_ext import build_ext
    use_cython = False


class build_ext_dependencies(build_ext):
    def finalize_options(self):
        build_ext.finalize_options(self)
        import numpy
        self.include_dirs.append(numpy.get_include())
        self.include_dirs.append("pygeodesic/geodesic_kirsanov")


sources = ["pygeodesic/geodesic.pyx"] if use_cython else ["pygeodesic/geodesic.cpp"]

setup(
    ext_modules=[
        Extension(
            "pygeodesic.geodesic",
            sources=sources,
            language="c++",
        )
    ],
    cmdclass={"build_ext": build_ext_dependencies},
)