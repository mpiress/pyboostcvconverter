// tutorial used: https://www.programmersought.com/article/1275785794/

#define PY_ARRAY_UNIQUE_SYMBOL pbcvt_ARRAY_API

#include <boost/python.hpp>
#include <iostream>
#include <pyboostcvconverter/pyboostcvconverter.hpp>

#include "HistologicalEntities.h"

namespace pbcvt {

using namespace boost::python;


//PyObject -> Vector
vector<cv::Mat> listTupleToVector_Int(PyObject* incoming) {
	vector<cv::Mat> data;
	
	for(Py_ssize_t i = 0; i < PyArray_Size(incoming); i++) {
		cv::Mat *value = pbcvt::fromNDArrayToMat(incoming, i);
		data.push_back(value);
	}
	
	return data;

}

//Vector -> PyObject
PyObject *vectorToListTuple_Int(PyObject *pyimg, PyObject *args) {
	PyObject* seq = PyList_New(bgr->size());
     
     int i = 0;
     for(std::vector<cv::Mat>::iterator it = bgr->begin() ; it != bgr->end(); ++it){
        PyObject* item = pbcvt::fromMatToNDArray(*it);
        PyList_SET_ITEM(seq, i, item);
        i++;
    }

    return seq;
}


static PyObject *segmentNucleiStg1Py(PyObject *pyimg, PyObject *args) {
    unsigned char blue;
    unsigned char green;
    unsigned char red;
    double T1;
    double T2;

    std::vector<cv::Mat> bgr;
    cv::Mat rbc;

    cv::Mat img;
    img = pbcvt::fromNDArrayToMat(pyimg);

    if (!PyArg_ParseTuple(args, "bbbdd", &blue, &green, &red, &T1, &T2)) {
        return NULL;
    }

    ::nscale::HistologicalEntities::segmentNucleiStg1(img, blue, green, red, T1, T2, &bgr, &rbc);

     

    return (seq1, pbcvt::fromMatToNDArray(*rbc))
}

#if (PY_VERSION_HEX >= 0x03000000)

static void *init_ar() {
#else
static void init_ar() {
#endif
    Py_Initialize();

    import_array();
    return NUMPY_IMPORT_ARRAY_RETVAL;
}

BOOST_PYTHON_MODULE(pbcvt) {
    // using namespace XM;
    init_ar();

    // initialize converters
    to_python_converter<cv::Mat, pbcvt::matToNDArrayBoostConverter>();
    matFromNDArrayBoostConverter();

    // expose module-level functions
    def("segmentNucleiStg1Py", segmentNucleiStg1Py);
}

}  // end namespace pbcvt
