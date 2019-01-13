# build a uniform split that contains N questions from each clevr family
# question families in the original clevr dataset are represented by a 0 - 89 index

import os
import argparse
import h5py
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--random_sample', default=0,
                    type=int, help='randomly sample questions')
parser.add_argument('--max_sample', default=2000,
                    type=int, help='size of split, effective only when random is true')
parser.add_argument('--n_questions_per_family', default=5, 
                    type=int, help='number of questions per family, effective when random is false')
parser.add_argument('--input_question_h5', default='../data/reason/clevr_h5/clevr_train_questions.h5',
                    type=str, help='path to input question h5 file')
parser.add_argument('--output_dir', default='../data/reason/clevr_h5',
                    type=str, help='output dir')
parser.add_argument('--output_index', default=None,
                    type=int, help='output repeat id')


def main(args):
    print('| importing questions from %s' % args.input_question_h5)
    input_questions = h5py.File(args.input_question_h5, 'r')
    N = len(input_questions['questions'])

    questions, programs, answers, question_families, orig_idxs, img_idxs = [], [], [], [], [], []
    family_count = np.zeros(90)

    if args.random_sample:
        max_sample = args.max_sample
        if args.output_index is not None:
            filename = 'clevr_train_%d_questions_%02d.h5' % (max_sample, args.output_index)
        else:
            filename = 'clevr_train_%d_questions.h5' % max_sample
        print('| randomly sampling %d questions' % max_sample)
    else:
        max_sample = args.n_questions_per_family * 90
        if args.output_index is not None:
            filename = 'clevr_train_%dquestions_per_family_%02d.h5' \
                       % (args.n_questions_per_family, args.output_index)
        else:
            filename = 'clevr_train_%dquestions_per_family.h5' \
                       % args.n_questions_per_family
        print('| drawing questions, %d per family' % args.n_questions_per_family)

    sample_seq = np.random.permutation(N)
    for i in sample_seq:
        fam_idx = int(input_questions['question_families'][i])
        if args.random_sample or family_count[fam_idx] < args.n_questions_per_family:
            family_count[fam_idx] += 1
            questions.append(input_questions['questions'][i])
            programs.append(input_questions['programs'][i])
            answers.append(input_questions['answers'][i])
            question_families.append(input_questions['question_families'][i])
            orig_idxs.append(input_questions['orig_idxs'][i])
            img_idxs.append(input_questions['image_idxs'][i])
        if family_count.sum() >= max_sample:
            break

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    output_file = os.path.join(args.output_dir, filename)

    print('sampled question family distribution')
    print(family_count)
    print('| saving output file to %s' % output_file)
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('questions', data=np.asarray(questions, dtype=np.int32))
        f.create_dataset('programs', data=np.asarray(programs, dtype=np.int32))
        f.create_dataset('answers', data=np.asarray(answers))
        f.create_dataset('image_idxs', data=np.asarray(img_idxs))
        f.create_dataset('orig_idxs', data=np.asarray(orig_idxs))
        f.create_dataset('question_families', data=np.asarray(question_families))

    print('| finish')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)