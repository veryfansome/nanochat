import os

from sandbox.tok_words_eval import benchmark_tokenizer_dir_name, get_base_dir, get_tokenizer, tokenized_for_comparison

test_texts = (
    """
WASHINGTON — Teeing up a blockbuster ruling, the Supreme Court on Friday agreed to decide the lawfulness of President Donald Trump’s contentious plan to roll back automatic birthright citizenship for nearly anyone born in the United States.

The eventual ruling in a case from New Hampshire, expected by the end of June, will likely determine conclusively whether Trump’s ambitious proposal can move forward.

The case sets up a major clash between a president whose aggressive use of executive power has been a defining characteristic of his second term and a court with a 6-3 conservative majority that has so far mostly avoided direct clashes with the White House.

"The Trump Administration looks forward to making its case on the issue of birthright citizenship on behalf of the American people," Abigail Jackson, a White House spokeswoman said in a statement.

Birthright citizenship has long been understood to be required under the Constitution’s 14th Amendment, which states: “All persons born or naturalized in the United States, and subject to the jurisdiction thereof, are citizens of the United States.”

The language was included in the constitutional amendment enacted after the Civil War to ensure that Black former slaves and their children were recognized as citizens.

Legal scholars of all ideological stripes have generally assumed the phrase to be self-explanatory, with the only exceptions being people born to foreign diplomats, invading hostile forces and members of some Native American tribes.

But Trump, as part of his immigration crackdown, has sought to unravel that historical understanding, embracing a hitherto fringe theory pushed by anti-immigration activists.

"For over 150 years, it has been the law and our national tradition that everyone born on U.S. soil is a citizen from birth," Cecillia Wang, national legal director of the American Civil Liberties Union, which is involved in the challenge, said in a statement. "We look forward to putting this issue to rest once and for all."

Under the administration’s view, birthright citizenship would be limited to those who have at least one parent who is a U.S. citizen or permanent legal resident. In that scenario, the right would not apply to babies born to temporary visitors who entered the country legally or to people who entered the country illegally.

The administration’s legal argument, presented by Solicitor General D. John Sauer, is that the “subject to the jurisdiction thereof” language confers citizenship upon only children who are not just present in the United States but also bear allegiance to it.

It is not enough merely to be subject to U.S. law, which is how the clause has traditionally been interpreted, he argues.
""".strip(),
)


if __name__ == '__main__':
    benchmark_tokenizer = (
        get_tokenizer(benchmark_tokenizer_dir_name)
        if os.path.isdir(os.path.join(get_base_dir(), benchmark_tokenizer_dir_name))
        else None
    )
    for test in test_texts:
        benchmark_tokens, tokens = tokenized_for_comparison(test, benchmark=benchmark_tokenizer)
        print("---")
        print(f"text: {test}")
        if benchmark_tokens:
            print(f"\nbenchmark: {len(benchmark_tokens)}")
            print(f"benchmark: {benchmark_tokens}")
        print(f"\ntokens: {len(tokens)}")
        print(f"tokens: {tokens}")
