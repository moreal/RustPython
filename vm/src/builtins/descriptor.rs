use rustpython_common::lock::PyRwLock;
use super::{PyStr, PyType, PyTypeRef};
use crate::class::PyClassImpl;
use crate::function::Either;
use crate::types::{Constructor, GetDescriptor, Unconstructible};
use crate::{AsObject, Context, Py, PyObjectRef, PyPayload, PyRef, PyResult, VirtualMachine};

#[derive(Debug)]
pub struct DescrObject {
    pub typ: PyTypeRef,
    pub name: String,
    pub qualname: PyRwLock<Option<String>>,
}

#[derive(Debug)]
pub enum MemberKind {
    ObjectEx = 16,
}

pub struct MemberDef {
    pub name: String,
    pub kind: MemberKind,
    pub getter_or_offset: Either<fn(PyObjectRef, &VirtualMachine) -> PyResult, usize>,
    pub doc: Option<String>,
}

impl std::fmt::Debug for MemberDef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemberDef")
            .field("name", &self.name)
            .field("kind", &self.kind)
            .field("doc", &self.doc)
            .finish()
    }
}

#[pyclass(name = "member_descriptor", module = false)]
#[derive(Debug)]
pub struct MemberDescrObject {
    pub common: DescrObject,
    pub member: MemberDef,
}

impl PyPayload for MemberDescrObject {
    fn class(vm: &VirtualMachine) -> &'static Py<PyType> {
        vm.ctx.types.member_descriptor_type
    }
}

fn calculate_qualname(descr: &DescrObject, vm: &VirtualMachine) -> PyResult<Option<String>> {
    let type_qualname = vm.get_attribute_opt(descr.typ.to_owned().into(), "__qualname__")?;
    match type_qualname {
        None => Ok(None),
        Some(obj) => {
            let str = obj.downcast::<PyStr>().map_err(|_| vm.new_type_error(
                "<descriptor>.__objclass__.__qualname__ is not a unicode object".to_owned(),
            ))?;
            Ok(Some(format!("{}.{}", str, descr.name)))
        }
    }
}

#[pyclass(with(GetDescriptor, Constructor), flags(BASETYPE))]
impl MemberDescrObject {
    #[pymethod(magic)]
    fn repr(zelf: PyRef<Self>) -> String {
        format!(
            "<member '{}' of '{}' objects>",
            zelf.common.name,
            zelf.common.typ.name(),
        )
    }

    #[pyproperty(magic)]
    fn doc(zelf: PyRef<Self>) -> Option<String> {
        zelf.member.doc.to_owned()
    }

    #[pyproperty(magic)]
    fn qualname(&self, vm: &VirtualMachine) -> PyResult<Option<String>> {
        if self.common.qualname.read().is_none() {
            *self.common.qualname.write() = calculate_qualname(&self.common, vm)?;
        }

        Ok(self.common.qualname.read().to_owned())
    }
}

// PyMember_GetOne
fn get_slot_from_object(
    obj: PyObjectRef,
    offset: usize,
    member: &MemberDef,
    vm: &VirtualMachine,
) -> PyResult {
    match member.kind {
        MemberKind::ObjectEx => match obj.get_slot(offset) {
            Some(obj) => Ok(obj),
            None => Err(vm.new_attribute_error(format!(
                "'{}' object has no attribute '{}'",
                obj.class().name(),
                member.name
            ))),
        },
    }
}

impl Unconstructible for MemberDescrObject {}

impl GetDescriptor for MemberDescrObject {
    fn descr_get(
        zelf: PyObjectRef,
        obj: Option<PyObjectRef>,
        _cls: Option<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult {
        match obj {
            Some(x) => {
                let zelf = Self::_zelf(zelf, vm)?;
                match zelf.member.getter_or_offset {
                    Either::A(getter) => (getter)(x, vm),
                    Either::B(offset) => get_slot_from_object(x, offset, &zelf.member, vm),
                }
            }
            None => Ok(zelf),
        }
    }
}

pub fn init(context: &Context) {
    let member_descriptor_type = &context.types.member_descriptor_type;
    MemberDescrObject::extend_class(context, member_descriptor_type);
}
