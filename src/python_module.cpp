// tutorial used: https://www.programmersought.com/article/1275785794/

#define PY_ARRAY_UNIQUE_SYMBOL pbcvt_ARRAY_API

#include <boost/python.hpp>
#include <iostream>
#include <pyboostcvconverter/pyboostcvconverter.hpp>
#include <HistologicalEntities.h>


namespace pbcvt {

using namespace boost::python;

//PyObject -> Vector
std::vector<cv::Mat> PyObjectToVector(PyObject *incoming) {
	std::vector<cv::Mat> data;
	
	for(Py_ssize_t i=0; i<PyList_Size(incoming); i++) {
		PyObject *value = PyList_GetItem(incoming, i);
		cv::Mat vet = pbcvt::fromNDArrayToMat(value);
		data.push_back(vet);
	}
	
	return data;
}

//Vector -> PyObject
PyObject* VectorToPyObject(vector<cv::Mat> bgr) {
	PyObject* seq = PyList_New(bgr.size());
     
     int i = 0;
     for(std::vector<cv::Mat>::iterator it = bgr.begin() ; it != bgr.end(); ++it){
        PyObject* item = pbcvt::fromMatToNDArray(*it);
        PyList_SET_ITEM(seq, i, item);
        i++;
    }

    return seq;
}


static PyObject *segmentNucleiStg1Py(PyObject *self, PyObject *args) {
    PyObject* result = PyList_New(2);
    unsigned char blue;
    unsigned char green;
    unsigned char red;
    double T1;
    double T2;

    std::vector<cv::Mat> bgr;
    cv::Mat rbc;

    cv::Mat img = pbcvt::fromNDArrayToMat(self);

    if (!PyArg_ParseTuple(args, "bbbdd", &blue, &green, &red, &T1, &T2)) {
        return NULL;
    }

    /*int HistologicalEntities::segmentNucleiStg1(
    const Mat &img, unsigned char blue, unsigned char green, unsigned char red,
    double T1, double T2, std::vector<Mat> *bgr, Mat *rbc,
    ::cciutils::SimpleCSVLogger *logger,
    ::cciutils::cv::IntermediateResultHandler *iresHandler)*/
    ::nscale::HistologicalEntities::segmentNucleiStg1(img, blue, green, red, T1, T2, &bgr, &rbc, NULL, NULL);
    
    PyObject *py_rbc = pbcvt::fromMatToNDArray(rbc);
    PyObject *py_bgr = VectorToPyObject(bgr);
    PyList_SET_ITEM(result, 0, py_rbc);
    PyList_SET_ITEM(result, 1, py_bgr); 

    return result;
}

static PyObject *segmentNucleiStg2Py(PyObject *self, PyObject *args) {
	PyObject* result = PyList_New(4);
	int reconConnectivity;

	std::vector<cv::Mat> bgr = PyObjectToVector(self);
    
    cv::Mat rc; 
	cv::Mat rc_recon;
	cv::Mat rc_open;

	if (!PyArg_ParseTuple(args, "i", &reconConnectivity)) {
        return NULL;
    }

	/*int HistologicalEntities::segmentNucleiStg2(
    int reconConnectivity, std::vector<Mat> *bgr, Mat *rc, Mat *rc_recon,
    Mat *rc_open, ::cciutils::SimpleCSVLogger *logger,
    ::cciutils::cv::IntermediateResultHandler *iresHandler)*/
	::nscale::HistologicalEntities::segmentNucleiStg2(reconConnectivity, &bgr, &rc, &rc_recon, &rc_open, NULL, NULL);

	PyObject *py_bgr      = VectorToPyObject(bgr);
	PyObject *py_rc       = pbcvt::fromMatToNDArray(rc);
	PyObject *py_rc_recon = pbcvt::fromMatToNDArray(rc_recon);
	PyObject *py_rc_open  = pbcvt::fromMatToNDArray(rc_open);
	PyList_SET_ITEM(result, 0, py_bgr);
	PyList_SET_ITEM(result, 1, py_rc);
	PyList_SET_ITEM(result, 2, py_rc_recon);
    PyList_SET_ITEM(result, 3, py_rc_open);

    return result; 

}

static PyObject *segmentNucleiStg3Py(PyObject *self, PyObject *args) {
	PyObject* result = PyList_New(5);
	int fillHolesConnectivity; 
	unsigned char G1;

	cv::Mat rc       = pbcvt::fromNDArrayToMat(PyList_GetItem(self, 0));
	cv::Mat rc_recon = pbcvt::fromNDArrayToMat(PyList_GetItem(self, 1));
    cv::Mat rc_open  = pbcvt::fromNDArrayToMat(PyList_GetItem(self, 2));
    
    cv::Mat bw1; 
    cv::Mat diffIm;

    if (!PyArg_ParseTuple(args, "ib", &fillHolesConnectivity, &G1)) {
        return NULL;
    }

	/*HistologicalEntities::segmentNucleiStg3(
    int fillHolesConnectivity, unsigned char G1, Mat *rc, Mat *rc_recon,
    Mat *rc_open, Mat *bw1, Mat *diffIm, ::cciutils::SimpleCSVLogger *logger,
    ::cciutils::cv::IntermediateResultHandler *iresHandler)*/
    ::nscale::HistologicalEntities::segmentNucleiStg3(fillHolesConnectivity, G1, &rc, &rc_recon, &rc_open, &bw1, &diffIm, NULL, NULL);

    PyObject *py_rc       = pbcvt::fromMatToNDArray(rc);
    PyObject *py_rc_recon = pbcvt::fromMatToNDArray(rc_recon);
    PyObject *py_rc_open  = pbcvt::fromMatToNDArray(rc_open);
    PyObject *py_bw1      = pbcvt::fromMatToNDArray(bw1);
	PyObject *py_diffIm   = pbcvt::fromMatToNDArray(diffIm);
	PyList_SET_ITEM(result, 0, py_rc);
	PyList_SET_ITEM(result, 1, py_rc_recon);
	PyList_SET_ITEM(result, 2, py_rc_open);
	PyList_SET_ITEM(result, 3, py_bw1);
    PyList_SET_ITEM(result, 4, py_diffIm);

    return result;
}

static PyObject *segmentNucleiStg4Py(PyObject *self, PyObject *args) {
	PyObject* result = PyList_New(2);
	int minSize; 
	int maxSize;

	cv::Mat bw1  = pbcvt::fromNDArrayToMat(self);

	cv::Mat bw1_t;

	if (!PyArg_ParseTuple(args, "ii", &minSize, &maxSize)) {
        return NULL;
    }

	/*int HistologicalEntities::segmentNucleiStg4(
    int minSize, int maxSize, Mat *bw1, Mat *bw1_t,
    ::cciutils::SimpleCSVLogger *logger,
    ::cciutils::cv::IntermediateResultHandler *iresHandler)*/
    ::nscale::HistologicalEntities::segmentNucleiStg4(minSize, maxSize, &bw1, &bw1_t, NULL, NULL);

    PyObject *py_bw1    = pbcvt::fromMatToNDArray(bw1);
    PyObject *py_bw1_t  = pbcvt::fromMatToNDArray(bw1_t);
    PyList_SET_ITEM(result, 0, py_bw1);
	PyList_SET_ITEM(result, 1, py_bw1_t);

    return result;
}

static PyObject *segmentNucleiStg5Py(PyObject *self, PyObject *args) {
	PyObject* result = PyList_New(4);
	unsigned char G2;

	cv::Mat diffIm = pbcvt::fromNDArrayToMat(PyList_GetItem(self, 0));
	cv::Mat bw1_t 	= pbcvt::fromNDArrayToMat(PyList_GetItem(self, 1));
	cv::Mat rbc    = pbcvt::fromNDArrayToMat(PyList_GetItem(self, 2));

	cv::Mat seg_open;
	
	if (!PyArg_ParseTuple(args, "b", &G2)) {
        return NULL;
    }

	/*int HistologicalEntities::segmentNucleiStg5(
    unsigned char G2, Mat *diffIm, Mat *bw1_t, Mat *rbc, Mat *seg_open,
    ::cciutils::SimpleCSVLogger *logger,
    ::cciutils::cv::IntermediateResultHandler *iresHandler)*/
    ::nscale::HistologicalEntities::segmentNucleiStg5(G2, &diffIm, &bw1_t, &rbc, &seg_open, NULL, NULL);

	PyObject *py_diffIm    = pbcvt::fromMatToNDArray(diffIm);   
	PyObject *py_bw1_t     = pbcvt::fromMatToNDArray(bw1_t);   
	PyObject *py_rbc       = pbcvt::fromMatToNDArray(rbc);   
	PyObject *py_seg_open  = pbcvt::fromMatToNDArray(seg_open); 
	PyList_SET_ITEM(result, 0, py_diffIm);
	PyList_SET_ITEM(result, 1, py_bw1_t);
	PyList_SET_ITEM(result, 2, py_rbc);
	PyList_SET_ITEM(result, 3, py_seg_open);  
	
	return result; 
}

static PyObject *segmentNucleiStg6Py(PyObject *self, PyObject *args) {
	PyObject* result = PyList_New(2);
	int minSizePl; 
	int watershedConnectivity;
	
	cv::Mat img      = pbcvt::fromNDArrayToMat(PyList_GetItem(self, 0));
	cv::Mat seg_open = pbcvt::fromNDArrayToMat(PyList_GetItem(self, 1));

	cv::Mat seg_nonoverlap;

	if (!PyArg_ParseTuple(args, "ii", &minSizePl, &watershedConnectivity)) {
        return NULL;
    }

	/*int HistologicalEntities::segmentNucleiStg6(
    const Mat &img, int minSizePl, int watershedConnectivity, Mat *seg_open,
    Mat *seg_nonoverlap, ::cciutils::SimpleCSVLogger *logger,
    ::cciutils::cv::IntermediateResultHandler *iresHandler)*/
    ::nscale::HistologicalEntities::segmentNucleiStg6(img, minSizePl, watershedConnectivity, &seg_open, &seg_nonoverlap, NULL, NULL);

    PyObject *py_seg_open        = pbcvt::fromMatToNDArray(seg_open);   
	PyObject *py_seg_nonoverlap  = pbcvt::fromMatToNDArray(seg_nonoverlap); 
	PyList_SET_ITEM(result, 0, py_seg_open);
	PyList_SET_ITEM(result, 1, py_seg_nonoverlap);

	return result;
}

static PyObject *segmentNucleiStg7Py(PyObject *self, PyObject *args) {
	PyObject* result = PyList_New(2);
	int minSizeSeg; 
	int maxSizeSeg; 
	int fillHolesConnectivity;

	cv::Mat seg_nonoverlap = pbcvt::fromNDArrayToMat(self);

	cv::Mat output;

    if (!PyArg_ParseTuple(args, "iii", &minSizeSeg, &maxSizeSeg, &fillHolesConnectivity)) {
        return NULL;
    }
	/*int HistologicalEntities::segmentNucleiStg7(
    cv::Mat *output, int minSizeSeg, int maxSizeSeg, int fillHolesConnectivity,
    Mat *seg_nonoverlap, ::cciutils::SimpleCSVLogger *logger,
    ::cciutils::cv::IntermediateResultHandler *iresHandler)*/
	::nscale::HistologicalEntities::segmentNucleiStg7(&output, minSizeSeg, maxSizeSeg, fillHolesConnectivity, &seg_nonoverlap, NULL, NULL);
	
	PyObject *py_seg_nonoverlap  = pbcvt::fromMatToNDArray(seg_nonoverlap);   
	PyObject *py_output          = pbcvt::fromMatToNDArray(output); 
	PyList_SET_ITEM(result, 0, py_seg_nonoverlap);
	PyList_SET_ITEM(result, 1, py_output);

	return result;
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
    def("segmentNucleiStg2Py", segmentNucleiStg2Py);
    def("segmentNucleiStg3Py", segmentNucleiStg3Py);
    def("segmentNucleiStg4Py", segmentNucleiStg4Py);
    def("segmentNucleiStg5Py", segmentNucleiStg5Py);
    def("segmentNucleiStg6Py", segmentNucleiStg6Py);
    def("segmentNucleiStg7Py", segmentNucleiStg7Py);
}

}  // end namespace pbcvt
