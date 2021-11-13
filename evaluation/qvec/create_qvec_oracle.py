# zcat finnish_vocab.txt.gz | awk '{$1=$1;print}' | cut -d' ' -f 2 | head -n 10000 | python create_qvec_oracle.py > morhps.fi

import json
import sys
from voikko import libvoikko


def main():
    included_keys = ['NUMBER', 'PERSON', 'MOOD', 'PARTICIPLE', 'COMPARISON', 'POSSESSIVE', 'TENSE']
    voikko = libvoikko.Voikko("fi")
    seen = set()
    for token in sys.stdin:
        token = token.strip().lower()
        if token in seen:
            continue
        seen.add(token)

        features = {}
        analyses = voikko.analyze(token)
        N = len(analyses)
        for analysis in analyses:
            for key in included_keys:
                if key in analysis:
                    fname = key.lower() + '.' + analysis[key]
                    features[fname] = features.get(fname, 0.0) + 1/N

        if len(features) > 1:
            sys.stdout.write(f'{token}\t{json.dumps(features)}\n')


if __name__ == '__main__':
    main()
