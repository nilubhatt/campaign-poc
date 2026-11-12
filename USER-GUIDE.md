# Campaign Intelligence: a guide for marketers

Campaign Intelligence is your team's memory of past campaigns: the decks, the photos, and
how each one actually did. When a new brief comes in, Claude checks it against that memory
and tells you what to keep, what to change and why, pointing at the past campaigns it is
comparing against.

You don't need to learn any commands. You talk to Claude in ordinary words, and Claude
does the rest.

---

## Before you start

Someone on your team installs it once on your Mac (it takes a few minutes). After that:

1. Open **Claude** (the desktop app).
2. Ask: **"Is Campaign Intelligence working?"**

Claude should answer that everything is working. If it doesn't seem to know what you mean,
see [If something isn't working](#if-something-isnt-working).

---

## How it works, in one minute

- **You talk, Claude does the work.** Say what you want in your own words. You don't need
  to name anything technical.
- **Nothing is saved without your OK.** Before Claude adds anything to the library, it shows
  you what it is about to save. Check it, correct anything that's wrong, then say **"yes,
  save it"**.
- **To add a deck or photo, give Claude the file's location; don't drag it in.** When you
  drag a file into the chat, Claude reads its text but Campaign Intelligence never gets the file itself, so
  comments, speaker notes and images are lost. Instead, copy the file's location and paste it
  into your message (see [Copying a file's location](#copying-a-files-location)).
- **The library starts empty.** Claude can only compare a new brief with campaigns you have
  added. You need **at least two past campaigns** before it can compare anything, and the
  more you add (with their results), the better its advice gets.

---

## The everyday jobs

### 1. Add a past campaign

Copy the deck's location (PowerPoint or PDF) and say:

> "Add this deck to the campaign library: /Users/sam/Downloads/Spring Launch.pptx.
> It ran in the US, spring 2026. Tags: leggings, social."

Claude shows you what it read and asks you to confirm. Say yes.

**Tips**
- **Give the file's location** rather than dragging the file in or pasting its text. Only
  the file carries the comments and notes people left on it, and only the file lets Claude
  check whether its images were used in another campaign. When Claude confirms, it says
  whether it read the comments. If it says it couldn't, give it the location again.
- Say **where it ran** (the market) and anything you'd want to search by later. You can add
  or fix these later too: *"Change the market on Spring Leggings Launch to UK."*

### 2. Add how it did

Once you know the results, tell Claude:

> "Spring Leggings Launch finished: engagement rate 5.1%, 17,200 new VIPs."

Again, Claude shows you what it will save and you confirm. **This is the most valuable
thing you can give the library.** A campaign without results can be found, but it can't
show whether that approach worked.

### 3. Find similar past campaigns

> "Have we done anything like a colourful activewear launch on social video?"

Claude lists the closest past campaigns, with how each one did when results are on file.
You can narrow it down: *"…only in the UK"*, *"…only ones that are finished"*.

### 4. Check a new brief

Drag the new brief into the chat (or paste its text) and say:

> "Check this brief against what we've done before."

Dragging is fine here, because the check works from the brief's text. To keep the brief in
the library as well, add it first as in job 1 and say it's a **proposed** campaign, then ask
Claude to check it.
You get back:

- a **verdict**: **approve**, **revise** or **reject**;
- a **one-line summary**;
- each **problem** it found, how serious it is, and **what to change**;
- for a *revise*, **what would make it an approve**.

Every point is backed by something you can check: a quote from a past campaign, or one of
your team's own written rules. Ask **"Why?"** about any point to see the evidence.

**Two kinds of problem, read them differently:**
- **"Breaks your rule"**: the brief goes against a rule your team wrote down. That isn't up
  for debate. Fix it, or change the rule.
- **"Done differently from a past campaign"**: the brief departs from something on file. That
  might be a mistake, or it might be a better idea. Claude asks rather than tells, and you
  decide.

### 5. Photos and creative

Photos work the same way: copy each photo's location (PNG, JPEG or WebP) into your message.

- **Add photos** of what actually ran: *"Add these photos to Spring Leggings Launch, they're
  from the event: /Users/sam/Pictures/event-1.jpg, /Users/sam/Pictures/event-2.jpg"*
- **Check for reuse:** *"Has this photo been used in another campaign?
  /Users/sam/Pictures/hero.png"*
- **Find a look:** *"Find past campaigns with photos that look like this one:
  /Users/sam/Pictures/hero.png"*

### 6. After a new campaign runs: close the loop

> "Summer Leggings Drop has finished: 9,800 new VIPs. How did our check hold up?"

Claude compares what it predicted with what happened, then asks you what the lesson was.
Your answer is what makes the next check better.

### 7. What needs my attention?

> "What's waiting on me?"

Claude gives you a short numbered list, such as a campaign with no results yet. Reply with
the number you want to deal with.

### 8. Where is the library thin?

> "Where is our library thin?" or "What's missing from the library?"

Claude tells you which markets or stages have few campaigns, and what to add first.

---

## Copying a file's location

1. Find the file in **Finder**.
2. **Right-click** it (or Control-click), then hold the **Option** key.
3. Choose **Copy "…" as Pathname**.
4. Paste it into your message to Claude (**⌘V**).

It looks like `/Users/sam/Downloads/Spring Launch.pptx`. Spaces in the name are fine.

---

## People's names and privacy

- The library is **stored on your Mac**, not in a shared online system. Claude reads what it
  needs from it to answer you, as it does with anything else in a chat.
- It records the names of people who **left comments or notes in a deck** you added, so it
  can tell a client's objection from your own team's note.
- To see what it holds about someone: *"What do we have on file about Jordan Avery?"*
- To take a name out: *"Remove Jordan Avery's name from the library."* The name is replaced
  everywhere. Their comments stay, but no longer say who wrote them.

---

## Good to know

- **Its advice is only as good as the library.** With only a few campaigns, or campaigns
  without results, treat a verdict as a well-sourced second opinion, not a final answer.
  Claude tells you when the evidence is thin.
- **It hasn't yet been checked against your clients' own verdicts.** Until it has, use its
  verdicts to start a conversation, not to end one.
- **Only written rules count as rules.** Claude flags a "breaks your rule" problem only for
  rules written into your team's rulebook. A guidelines deck you add to the library is used
  as background, not checked as rules. To add or change a rule, ask whoever installed it.

---

## If something isn't working

1. **Quit Claude completely and reopen it.** On a Mac press **⌘Q** (closing the window is not
   enough), then open Claude again.
2. Ask Claude: **"Run a Campaign Intelligence health check."** It tells you in plain words
   which part isn't working.
3. Still stuck? Send whoever installed it a screenshot of what Claude said.
