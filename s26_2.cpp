#include <Python.h>
#include <iostream>

int main() {
    // Initialize the Python interpreter
    Py_Initialize();

    // Import the Python script
    PyObject* pName = PyUnicode_FromString("s26");  // Module name without .py
    PyObject* pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule != nullptr) {
        // Get the reference to the Python function
        PyObject* pFunc = PyObject_GetAttrString(pModule, "add");

        if (pFunc && PyCallable_Check(pFunc)) {
            // Prepare arguments
            PyObject* pArgs = PyTuple_Pack(2, PyLong_FromLong(5), PyLong_FromLong(3));

            // Call the Python function
            PyObject* pValue = PyObject_CallObject(pFunc, pArgs);
            Py_DECREF(pArgs);

            if (pValue != nullptr) {
                // Print the result
                std::cout << "Result of add(5, 3): " << PyLong_AsLong(pValue) << std::endl;
                Py_DECREF(pValue);
            } else {
                PyErr_Print();
                std::cerr << "Call to add() failed." << std::endl;
            }

            Py_DECREF(pFunc);
        } else {
            if (PyErr_Occurred()) PyErr_Print();
            std::cerr << "Function 'add' not found or not callable." << std::endl;
        }

        Py_DECREF(pModule);
    } else {
        PyErr_Print();
        std::cerr << "Failed to load Python module." << std::endl;
    }

    // Finalize the Python interpreter
    Py_Finalize();

    return 0;
}



// set PYTHONHOME=I:\Installs
// set PYTHONPATH=I:\Installs\lib
// g++ -o s26_2 s26_2.cpp -I"I:\Installs\include" -L"I:\Installs\libs" -lpython39
// s26_2.exe