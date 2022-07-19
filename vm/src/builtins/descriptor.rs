use super::PyType;

pub struct PyDescrObject {
    type_obj: PyType,
    name: Option<String>,
    qualname: Option<String>,
}
