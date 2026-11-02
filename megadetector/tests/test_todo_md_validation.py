"""

Validates the format of TODO.md (the repo's work item list), as described at the top of
that file.

TODO.md is rendered by the Markdown Task Viewer (https://github.com/agentmorris/task-viewer),
which silently leaves out any work item that doesn't have a priority, an effort level, and a
tag, and silently drops description text that comes after those lines.  So this test parses
TODO.md the same way the task viewer does, and fails on anything the task viewer would leave
out or misread.

"""

#%% Imports

import os
import re


#%% Constants

todo_file = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..','TODO.md'))

# The patterns the task viewer uses to recognize the lines that follow each description
priority_pattern = re.compile(r'^P[0-4]$')
effort_pattern = re.compile(r'^E[0-4]$')
sort_weight_pattern = re.compile(r'^S-?\d+$')
tag_pattern = re.compile(r'^![a-zA-Z0-9\-_\[\]]+$')

# Special tags, which the task viewer doesn't count as tags
special_tags = ('name','also-see')


#%% Support functions

def _read_work_items(todo_text):
    """
    Split the "Issues" section of TODO.md into work items, the same way the task viewer
    does.

    Args:
        todo_text (str): the contents of TODO.md

    Returns:
        list: a list of (title, lines) tuples, one per work item
    """

    # The task viewer splits the file at every line that starts with "# ", and uses the
    # first section that starts with "Issues"
    sections = re.split(r'^# ',todo_text,flags=re.MULTILINE)
    issues_sections = [s for s in sections if s.strip().startswith('Issues')]
    assert len(issues_sections) > 0, 'No "# Issues" section in {}'.format(todo_file)
    issues_text = re.sub(r'^Issues\s*[\r\n]+','',issues_sections[0])

    # Split at level-2 headings; the result alternates between titles and contents
    tokens = re.split(r'^## (.+)$',issues_text,flags=re.MULTILINE)[1:]

    work_items = []
    for i_token in range(0,len(tokens),2):
        title = tokens[i_token].strip()
        lines = tokens[i_token+1].strip().split('\n')
        work_items.append((title,lines))

    return work_items


def _validate_work_item(title,lines):
    """
    Check one work item.

    Args:
        title (str): the work item's title
        lines (list): the lines of the work item, below the title

    Returns:
        list: a list of problems with this work item, empty if there are none
    """

    n_priorities = 0
    n_efforts = 0
    tags = []
    description_lines = []
    text_after_description = []

    in_description = True

    for line in lines:

        s = line.strip()

        if priority_pattern.match(s):
            n_priorities += 1
            in_description = False
        elif effort_pattern.match(s):
            n_efforts += 1
            in_description = False
        elif sort_weight_pattern.match(s):
            in_description = False
        elif tag_pattern.match(s):
            tag_name = re.match(r'^!([a-zA-Z0-9\-_]+)',s).group(1)
            if tag_name not in special_tags:
                tags.append(tag_name)
            in_description = False
        elif in_description:
            description_lines.append(s)
        elif len(s) > 0:
            text_after_description.append(s)

    # ...for each line

    problems = []

    if not any(len(s) > 0 for s in description_lines):
        problems.append('no description')
    if n_priorities != 1:
        problems.append('expected one priority line (P0 through P4), found {}'.format(
            n_priorities))
    if n_efforts != 1:
        problems.append('expected one effort line (E0 through E4), found {}'.format(
            n_efforts))
    if len(tags) == 0:
        problems.append('no tags (e.g. !feature)')
    for s in text_after_description:
        problems.append('text after the priority/effort/tag lines, which the task viewer '
                        'does not show: "{}"'.format(s))

    return ['{}: {}'.format(title,p) for p in problems]

# ...def _validate_work_item(...)


#%% Tests

def test_todo_md_format():
    """
    Check that every work item in TODO.md has a description followed by one priority, one
    effort level, and at least one tag, and that the task viewer sees every work item.
    """

    with open(todo_file,'r',encoding='utf-8') as f:
        todo_text = f.read()

    work_items = _read_work_items(todo_text)
    assert len(work_items) > 0, 'No work items found in {}'.format(todo_file)

    problems = []

    for title,lines in work_items:
        problems.extend(_validate_work_item(title,lines))

    # A line starting with "# " inside a work item ends the "Issues" section for the task
    # viewer, which then loses every work item after it
    n_headings = len(re.findall(r'^## ',todo_text.split('\n# Issues',1)[-1],flags=re.MULTILINE))
    if n_headings != len(work_items):
        problems.append('found {} level-2 headings after "# Issues", but the task viewer '
                        'would only see {} work items'.format(n_headings,len(work_items)))

    assert len(problems) == 0, \
        'Problems in {}:\n{}'.format(todo_file,'\n'.join(problems))

# ...def test_todo_md_format()


def test_todo_md_validation_catches_problems():
    """
    Check that the validation code catches the problems it's supposed to catch, using
    made-up work items.
    """

    def problems_for(item_text):
        lines = item_text.strip().split('\n')
        return _validate_work_item('item',lines)

    # A well-formed work item, including the optional sort weight and special tags
    assert problems_for('Do the thing.\n\n!name[the-thing]\n\nP1\n\nE2\n\n!feature\n!bug\nS-1') == []

    assert len(problems_for('P1\n\nE2\n\n!feature')) == 1
    assert len(problems_for('Do the thing.\n\nE2\n\n!feature')) == 1
    assert len(problems_for('Do the thing.\n\nP1\n\n!feature')) == 1
    assert len(problems_for('Do the thing.\n\nP1\n\nP2\n\nE2\n\n!feature')) == 1
    assert len(problems_for('Do the thing.\n\nP7\n\nE2\n\n!feature')) == 1

    # A special tag is not a tag
    assert len(problems_for('Do the thing.\n\nP1\n\nE2\n\n!name[the-thing]')) == 1

    # Text after the priority/effort/tag lines
    assert len(problems_for('Do the thing.\n\nP1\n\nE2\n\n!feature\n\nAlso this.')) == 1

    # A stray "# " line hides the work items after it
    todo_text = '# Issues\n\n## A\n\nDo A.\n\nP1\n\nE1\n\n!feature\n\n' + \
                '# Not a heading\n\n## B\n\nDo B.\n\nP1\n\nE1\n\n!feature\n'
    assert [title for title,_ in _read_work_items(todo_text)] == ['A']

# ...def test_todo_md_validation_catches_problems()
