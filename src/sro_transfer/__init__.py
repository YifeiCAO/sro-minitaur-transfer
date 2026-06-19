"""SRO x Minitaur cross-task individual-transfer study.

Pipeline (see README and the experiment plan):
  Phase 0a  data -> natural language        (sro_transfer.data)
  Phase 0b  reliability ceiling             (sro_transfer.diagnostics.reliability)
  Phase 0c  handcrafted transfer matrix     (sro_transfer.diagnostics.transfer_matrix)
  Phase 1   M_pop population fine-tune       (sro_transfer.model.mpop)
  Phase 2   person-encoder + injection       (sro_transfer.model.transfer_model)
  Phase 3   cross-task identification        (sro_transfer.eval.identification)
  Phase 4   full model + NLL ablations       (sro_transfer.eval.nll)
"""
__version__ = "0.1.0"
