# Mechanism Endpoint Contract

MECHANISM_ENDPOINT_CONTRACT = AUDIT_ONLY

Required future endpoint before training:
- label-free separation between harmful undersegmentation and retained positive v2 units;
- no use of case id, GT, fixed slice index, test, or singleton threshold;
- explicit candidate-vs-v2 retained strength checks;
- endpoint logs proving compensation path activates only in supported risk units.
