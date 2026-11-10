"""One explicit feedback/channel profile shared by training and frozen evaluation."""
from dataclasses import replace
from coding import Config

ARCHITECTURES = {
    'hd_real': {'feedback_link': 'half_duplex', 'feedback_ideal': False},
    'hd_ideal': {'feedback_link': 'half_duplex', 'feedback_ideal': True},
    'ded_real': {'feedback_link': 'dedicated', 'feedback_ideal': False},
    'ded_ideal': {'feedback_link': 'dedicated', 'feedback_ideal': True},
}


def add_feedback_arguments(parser):
    cfg = Config()
    parser.add_argument('--feedback-rate-mbps', type=float, default=cfg.feedback_rate_bps/1e6,
                        help='Dedicated real link rate in Mbit/s')
    parser.add_argument('--forward-propagation-us', type=float, default=cfg.forward_propagation_s*1e6)
    parser.add_argument('--feedback-propagation-us', type=float, default=cfg.feedback_propagation_s*1e6)
    parser.add_argument('--feedback-processing-us', type=float, default=cfg.feedback_processing_s*1e6,
                        help='Aggregate fixed latency lumped after feedback arrival; not a CPU model')
    parser.add_argument('--feedback-loss-probability', type=float, default=cfg.feedback_loss_probability,
                        help='Independent report loss probability (real feedback only)')
    parser.add_argument('--turnaround-us', type=float, default=cfg.turnaround_s*1e6,
                        help='Half-duplex direction-change gap (SIFS), paid twice per report')
    parser.add_argument('--epoch-rf-packets', type=int, default=cfg.epoch_rf_packets,
                        help='Weather epoch in RF packet durations (declared main scenario: 10)')


def config_from_arguments(args, architecture=None):
    cfg = replace(Config(), feedback_rate_bps=args.feedback_rate_mbps*1e6,
                  forward_propagation_s=args.forward_propagation_us*1e-6,
                  feedback_propagation_s=args.feedback_propagation_us*1e-6,
                  feedback_processing_s=args.feedback_processing_us*1e-6,
                  epoch_rf_packets=args.epoch_rf_packets,
                  feedback_loss_probability=args.feedback_loss_probability,
                  turnaround_s=args.turnaround_us*1e-6,
                  **(ARCHITECTURES[architecture] if architecture else {}))
    cfg.validate()
    return cfg


def feedback_metadata(cfg):
    if cfg.feedback_link == 'half_duplex':
        timing = ('ideal: no airtime or RF interruption; usable at t+d_f at an RF boundary'
                  if cfg.feedback_ideal else
                  'real: forward RF silent t..t+d_f+T+F/R_R+d_b+T (stop and wait, turnaround T); '
                  'usable at arrival + processing')
    else:
        timing = ('ideal: usable at snapshot (zero serialization, propagation, processing)'
                  if cfg.feedback_ideal else
                  'real: snapshot + FIFO wait + F/R_FB + one-way propagation + processing')
    return {'link': cfg.feedback_link, 'ideal': cfg.feedback_ideal,
            'rate_bps': cfg.rates[1] if cfg.feedback_link == 'half_duplex' else cfg.feedback_rate_bps,
            'header': 'first completion-position index; fixed pre-shared block/session context',
            'header_bits': cfg.feedback_header_bits, 'payload_bits': cfg.window,
            'report_bits': cfg.feedback_report_bits,
            'forward_propagation_s': cfg.forward_propagation_s,
            'feedback_propagation_s': cfg.feedback_propagation_s,
            'aggregate_processing_s': cfg.feedback_processing_s, 'timing': timing,
            'turnaround_s': cfg.turnaround_s,
            'turnaround_source': ('direction-change gap = DMG aSIFSTime = 3 us, IEEE Std 802.11-2024 '
                                  'Table 20-30 (60 GHz DMG PHY); by Eq. (10-2) SIFS = aRxPHYDelay + '
                                  'aMACProcessingDelay + aRxTxTurnaroundTime; paid twice per report'),
            'forward_RF_interrupted': cfg.feedback_link == 'half_duplex', 'feedback_erasure': cfg.feedback_loss_probability,
            'feedback_loss_model': ('independent per report: a frame failing its error check (e.g. CRC) '
                                    'is discarded, so p stands in for frame protection without modeling '
                                    'check/coding bits; airtime still used; gap detected from the next '
                                    'position header; outcomes of a lost report are never confirmed and '
                                    'its packets stay in the prediction under the unreported-packet rule'),
            'association': 'completion-order positions; ties FSO before RF; synchronized erasure slots',
            'excluded': ['block setup', 'preamble/FEC/CRC on reliable service', 'contention',
                         'radio turnaround guard time', 'clock drift', 'forward decoder compute',
                         'terminal STOP']}
