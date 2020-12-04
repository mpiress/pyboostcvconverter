//tutorial used: https://www.programmersought.com/article/1275785794/

#define PY_ARRAY_UNIQUE_SYMBOL pbcvt_ARRAY_API

#include <boost/python.hpp>
#include <pyboostcvconverter/pyboostcvconverter.hpp>
#include <iostream>

namespace pbcvt {

    using namespace boost::python;

    static PyObject *segmentNucleiStg1Py(PyObject *pyimg, PyObject *args) {
        unsigned char blue;
        unsigned char green;
        unsigned char red;
        double T1;
        double T2;

        std::vector<cv::Mat> *bgr;
        cv::Mat *rbc;

        cv::Mat img;
        img = pbcvt::fromNDArrayToMat(pyimg);

        if (!PyArg_ParseTuple(args, "bbbdd", &blue, &green, &red, &T1, &T2)) {
            return NULL;
        }

        std::cout << "Read parameters " << blue << ", " << green << ", " << red
                  << ", " << T1 << ", " << T2 << ", "
                  << " and will return zero\n";

        
        //PyObject* seq1 = PyList_New(bgr->size());
        //int i = 0;
        //for(std::vector<cv::Mat>::iterator it = bgr->begin() ; it != bgr->end(); ++it){
        //    PyObject* item = pbcvt::fromMatToNDArray(*it);
        //    PyList_SET_ITEM(seq1, i, item);
        //    i++;
        //}

        //return seq1, pbcvt::fromMatToNDArray(*rbc);
        return 0;
    }


#if (PY_VERSION_HEX >= 0x03000000)

    static void *init_ar() {
#else
        static void init_ar(){
#endif
        Py_Initialize();

        import_array();
        return NUMPY_IMPORT_ARRAY_RETVAL;
    }

    BOOST_PYTHON_MODULE (pbcvt) {
        //using namespace XM;
        init_ar();

        //initialize converters
        to_python_converter<cv::Mat,pbcvt::matToNDArrayBoostConverter>();
        matFromNDArrayBoostConverter();

        //expose module-level functions
        def("segmentNucleiStg1Py", segmentNucleiStg1Py);
    }

} //end namespace pbcvt
