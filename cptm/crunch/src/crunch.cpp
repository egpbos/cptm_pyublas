#include "./crunch.hpp" 

using namespace crunch;

// Module ======================================================================
BOOST_PYTHON_MODULE(crunch)
{
  boost::python::def("test1", test1);
  boost::python::def("test2", test2);
  boost::python::def("p_x", p_x);
  boost::python::def("p_z", p_z);
  boost::python::def("sample_from", sample_from);
  boost::python::def("sample_from2", sample_from2);
  boost::python::def("sample_from3", sample_from3);
  boost::python::class_<Sampler>("Sampler");
  boost::python::class_<Sampler2>("Sampler2");
}
