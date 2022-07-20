use super::PyType;

pub struct PyDescrObject {
    type_obj: PyType,
    name: Option<String>,
    qualname: Option<String>,
}

#[pyclass(module = false, name = "wrapper_descriptor")]
pub struct PyWrapperDescrObject {
    common: PyDescrObject,

}