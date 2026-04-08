from graders.base_grader import BaseGrader

class HardGrader(BaseGrader):
    required_probes = {"friction", "alignment", "stiffness"}
