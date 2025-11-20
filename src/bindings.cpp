#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "GroupedRegressionTree.h"

namespace py = pybind11;

PYBIND11_MODULE(grouped_regression_tree, m) {
    py::class_<GroupedRegressionTree>(m, "GroupedRegressionTree")
        .def(py::init<int,int,int>())
        .def("fit", &GroupedRegressionTree::fit)
        .def("predict", &GroupedRegressionTree::predict)
        .def("predict_single", &GroupedRegressionTree::predict_single)
        .def("clone", &GroupedRegressionTree::clone)
        .def("__deepcopy__", [](const GroupedRegressionTree &self, py::dict) {
            return self.clone();
        })
        .def("export_tree", &GroupedRegressionTree::export_tree);
}
