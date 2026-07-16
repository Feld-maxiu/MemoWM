from residualmem.data import load_trajectory, save_trajectory
from residualmem.types import FieldSpec, StateSchema, Trajectory


def test_trajectory_storage_preserves_literal_and_typed_values(tmp_path):
    schema = StateSchema(
        "typed-data",
        (
            FieldSpec("flag", "bool"),
            FieldSpec("count", "integer"),
            FieldSpec("text", "literal"),
        ),
    )
    source = Trajectory(
        (
            schema.make_state((False, -2, "hello")),
            schema.make_state((True, 7, "world")),
        ),
        (3,),
        "typed",
    )
    path = tmp_path / "typed.npz"

    save_trajectory(path, source, schema)

    assert load_trajectory(path, schema) == source
