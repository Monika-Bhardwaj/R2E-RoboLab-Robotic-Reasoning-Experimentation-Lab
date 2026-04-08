from graders.base_grader import BaseGrader

class MediumGrader(BaseGrader):
    required_probes = {"friction", "alignment"}
