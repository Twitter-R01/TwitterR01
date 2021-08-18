**Purpose:** Provide training on PSC Bridges infrastructure and RITHM code repository. Attendees will learn basic data operations for the Twitter R01 project through instruction and practice. 
 - **Presenter:** Jason Colditz, MEd
 - **Location:** https://pitt.zoom.us/my/colditzjb


#### Sessions:
* [2020-09-04: Intro to Bridges / Data parsing](#2020-09-04)
* [2020-09-11: Keyword frequencies](#2020-09-11)
* [2020-09-18: Data subsampling](#2020-09-18)
* [2020-09-25: Missing data / Multiparsing](#2020-09-25)
* [2020-10-02: New developments / Projects](#2020-10-02)


<a id="2020-09-04"></a>
## 2020-09-04 Friday 2pm EST / 1pm CST 
Please follow along with each of these steps on your own computer.

**1. Get connected to Bridges**
   * Navigate to [Bridges OnDemand](https://ondemand.bridges.psc.edu/)
     * File manager is under: _Files_ \> _Home Directory_
     * Shell prompt is under: _Clusters_ \> _Bridges Shell Access_
   * Open a shell session for next steps
     * Note: Right-click will not paste, use Ctrl+V instead

**2. Quick tour of important Bridges locations**
   * There are no "shared" folders - everything lives inside user folders
     * `cd ~` Your home folder
     * `cd /home/jcolditz/twitter/RITHM/` RITHM master repository
   * All data are stored on "Pylon5", under our "be5fpap" group
     * `cd /pylon5/be5fpap/<user>/` Your Pylon5 data folder
     * `cd /pylon5/be5fpap/jcolditz/streamer_raw/` Raw Twitter data are here

**3. Quick review of BASH commands** 
   * Create a symlink: `ln -s /pylon5/be5fpap/<user> ~/data`
   * Check physical location: `pwd -P`
   * List file details in directory: `ls -la`
   * Set up group permissions: `chmod -R g=rx .`

**4. Clone the GitHub repository and do some housekeeping** 
```
cd /pylon5/be5fpap/<username>/

git clone https://github.com/CRMTH/UArk/

chmod -R g=rx .

ln -s /pylon5/be5fpap/<username>/UArk/recipes ~/recipes

cd ~/recipes/
```

**5. Get started with Hookah recipe**
   * Go to the [Wiki page](https://github.com/Twitter-R01/TwitterR01/wiki/Recipe-01:-Hookah)
   * Follow along with each of the steps

#### For next week: 
Have Hookah data for dates 20200101-20200831 in your local repo folders.

<a id="2020-09-11"></a>
## 2020-09-11 Friday 2pm EST / 1pm CST 
Please follow along with each of these steps on your own computer.

**1. Get connected to Bridges**
   * See Step 1 from last week 

**2. Check on last week's parser job**
```
cd ~/recipes/hookah/data/

ls -la

cd ../

clear

cat *.out

```

So many ascii/unicode errors! This is new and unexplained behavior - more on that later. Here's snippet of output to look at:

```
# FILE STATS #
FILE:       20200219000000_streamer.json
TWEETS:     583559
MATCHES:    14206
WARNINGS:   0
ERRORS:     0
DUPLICATES:  0
LARGE DUPLICATES:    0

20200220000000_streamer.json : HiMem failed! Trying with LoMem...

# FILE STATS #
FILE:       20200220000000_streamer.json
TWEETS:     617617
MATCHES:    68620
WARNINGS:   0
ERRORS:     7
DUPLICATES:  0
LARGE DUPLICATES:    0
```

With 7 errors, the 20200220 file might be a good one to focus on for testing/troubleshooting. Also, note this line of output, where HiMem failed and and fell back to LoMem mode:

`20200222000000_streamer.json : HiMem failed! Trying with LoMem...`

This is expected behavior when badly formed JSON objects cause HiMem to fail (e.g., the streamer process terminated abruptly with a half-written JSON object - it happens). But, we're seeing other types of errors here, for example:

```
Traceback (most recent call last):
  File "/home/jcolditz/twitter/RITHM/parser/decode.py", line 171, in fixjson
 decoder.writeToCSV(self, data, parsed_text, parsed_quote, fileName, count)
  File "/home/jcolditz/twitter/RITHM/parser/decode.py", line 468, in writeToCSV
 saveFile.writerow([entity for entity in entities])
UnicodeEncodeError: 'ascii' codec can't encode character '\xe9' in position 328: ordinal not in range(128)
```

This is an encoding error that I've not seen for years of working with these data, so I don't have a good explanation for why it's happening now. Note that the issue may be related to any number of text fields. "t\_text" and "t\_quote" are common culprits, but could also be in other fields where users enter non-English characters that aren't natively handled by ASCII codecs. However, I've been completely unable to reproduce the error after re-parsing that file half a dozen times (even using a much older version of the parser). So, let's take a minute to compare my output to your output, to see if you had the same output and errors when you parsed... 

Run this: 
```
cd ~/recipes/hookah/

cat *.out | grep -A 7 '20200219\|20200220'
```

Now here's my output when I re-ran it without any errors. Note that there are also fewer tweet "matches" for 20200220, though 20200219 was unaffected.

```
# FILE STATS #
FILE:       20200219000000_streamer.json
TWEETS:     583559
MATCHES:    14206
WARNINGS:   0
ERRORS:     0
DUPLICATES:  0
LARGE DUPLICATES:    0

# FILE STATS #
FILE:       20200220000000_streamer.json
TWEETS:     617618
MATCHES:    54671
WARNINGS:   0
ERRORS:     0
DUPLICATES:  0
LARGE DUPLICATES:    0
```

   * **Question:** Any hypotheses on why this happened? Multiple users accessing the same files and scripts at the same time _shouldn't_ be an issue, but that is the best possible explanation that I came up with. 


Also, this seems like a good time to start working with RITHM code base... 

**3. Set up a local RITHM repository and branch it**

```
cd ~

git clone https://github.com/CRMTH/RITHM/

git checkout fixascii

git branch

```

   * **Question:** Will you all be able to push updates to the "fixascii" remote branch that I created, or do you need to create your own branches? Let's discuss best practice (I don't know).


**4. Back to the Hookah recipe: keyword frequency counts**
   * Go to the [Wiki page](https://github.com/Twitter-R01/TwitterR01/wiki/Recipe-01:-Hookah)
   * Follow along with each of the steps

#### For next week: 
   * Should we re-parse and see what happens? (_note:_ remove old output files first or they will be appended to)
   * Have Hookah frequency output in your local repo folders.


<a id="2020-09-18"></a>
## 2020-09-18 Friday 2pm EST / 1pm CST 
Please follow along with each of these steps on your own computer.

**1. Get connected to Bridges**
   * See Step 1 from week 1 

**2. Check on the past parser job**

Let's first take a look to see if everyone has the same error counts as we discovered last week... 
```
cd ~/recipes/hookah/

cat *.out | grep -A 7 '20200219\|20200220'
```

Here's what it looks like when we found the errors: 

```
# FILE STATS #
FILE:       20200219000000_streamer.json
TWEETS:     583559
MATCHES:    14206
WARNINGS:   0
ERRORS:     0
DUPLICATES:  0
LARGE DUPLICATES:    0

20200220000000_streamer.json : HiMem failed! Trying with LoMem...

# FILE STATS #
FILE:       20200220000000_streamer.json
TWEETS:     617617
MATCHES:    68620
WARNINGS:   0
ERRORS:     7
DUPLICATES:  0
LARGE DUPLICATES:    0
```


Now here's my output when I re-ran it without any errors. Note that there are also fewer tweet "matches" for 20200220, though 20200219 was unaffected.

```
# FILE STATS #
FILE:       20200219000000_streamer.json
TWEETS:     583559
MATCHES:    14206
WARNINGS:   0
ERRORS:     0
DUPLICATES:  0
LARGE DUPLICATES:    0

# FILE STATS #
FILE:       20200220000000_streamer.json
TWEETS:     617618
MATCHES:    54671
WARNINGS:   0
ERRORS:     0
DUPLICATES:  0
LARGE DUPLICATES:    0
```

   * **Discuss:** How does this look for you all? Are the errors present? If so, then at least the errors are systematic (that bodes well for solving them). If not, then the errors were transient (hopefully they're no longer an issue, but it remains a mystery what caused them).


**3. Back to the Hookah recipe: data subsampling**
   * Go to the [Wiki page](https://github.com/Twitter-R01/TwitterR01/wiki/Recipe-01:-Hookah)
   * Follow along with each of the steps


**4. Discuss next steps**
   * What bugs did we find in the RITHM code? How can we fix them?
   * You now know how to use the basic scripts (parser.py, freq_out.py, and subsample.py). This is sufficient knowledge to get started with typical Aim 1 projects. Let's make use of this knowledge and begin work on Alex's tobacco/alcohol content area.


#### For next week: 
   * Submit GitHub issue(s) related to any bugs: use the UArk repo for now. Continue the debugging discussions there.
   * Have Hookah subsamples in your local repo folders.

<a id="2020-09-25"></a>
## 2020-09-25 Friday 2pm EST / 1pm CST 
Please follow along with each of these steps on your own computer.

**1. Get connected to Bridges**
   * See Step 1 from week 1 

**2. Check on the past parser job**

I ran the hookah parser one more time after including Will's RITHM update... 
```
cd ~/recipes/hookah/

cat *.out | grep -A 7 '20200219\|20200220'
```

The output matches what we found last week: 

```
# FILE STATS #
FILE:       20200219000000_streamer.json
TWEETS:     583559
MATCHES:    14206
WARNINGS:   0
ERRORS:     0
DUPLICATES:  0
LARGE DUPLICATES:    0

# FILE STATS #
FILE:       20200220000000_streamer.json
TWEETS:     617618
MATCHES:    54671
WARNINGS:   0
ERRORS:     0
DUPLICATES:  0
LARGE DUPLICATES:    0
```

   * **Discuss:** Are we confident that this is working consistently or do we want to troubleshoot further? Let's also check missing data.

**3. Checking for partial / missing data:**
   * There's a script (in development) for this ("/pylon5/be5fpap/jcolditz/scripts/partial\_data/partial\_data.py")
   * It's not battle-tested enough for RITHM inclusion yet; some of the functions should move to parselogic.py
   * Known issues:
     * Does not handle typical "99" as maximum calendar day
     * Some Twitter files are throwing errors

Run this:
```
cd ~/recipes/hookah/

sbatch partial_data.sb 
```

Selected output for "tweets" matches the parser "MATCHES" counts:
```
date	files	tweets	uptime	uptime_pct
...
20200219	1	14206	86383	100.0
20200220	1	54671	86397	100.0
...

```

**4. Onward to the Alcohol recipe:**
   * Go to the [Wiki page](https://github.com/Twitter-R01/TwitterR01/wiki/Recipe-03:-Vape-&-Alcohol)
   * Follow along with each of the steps for multiparsing


**5. Discuss next steps**
   * Any additional updates needed to parser at this point?
     * I updated the emoji function to resolve some badly formed output
     * "emojilist6.csv" is the most current emoji reference file (fixed capitalization)
   * Several actionable projects on the horizon:
     * Hookah + COVID (Brian Primack)
     * Tobacco/Vape + Alcohol (Alex Russell)
     * Vape + Policy (Page Dobbs)


#### For next week: 
   * Submit GitHub issue(s) related to any bugs: use the UArk repo for now. Continue the debugging discussions there.
   * Begin applying initial steps (parsing, frequencies, subsampling) to the Alcohol data
   * Get ready for the Hookah-COVID project (data span: 20200101-20200930)


<a id="2020-10-02"></a>
## 2020-10-02 Friday 2pm EST / 1pm CST 
Please follow along with each of these steps on your own computer.

**1. Get connected to Bridges**
   * See Step 1 from week 1 

**2. Stash / pull / etc. the RITHM repo**
   * Get the master branch updates (several new developments from last week).

**3. Run Hookah script**
   * This will create a "covid_project" subdirectory and populate some useful output.
   * **Don't forget to change folder names for each line of code (unless noted otherwise)!**

```
cd ~/recipes/hookah/

sbatch readysetgo.sb 
```

**4. Missing data developments**
   * Here: https://github.com/CRMTH/UArk/issues/2
   * Read the full paper at your leisure

   * **Discuss:** Known data capture limitations and considerations. What's next for RITHM?

**5. Determine next steps**
 1. Who is the primary contact while Jason is out?
 2. Should we move to bi-weekly meetings or keep them weekly?
  * Specific "seminar" topics or free-form Q&A?
 3. Several actionable projects now:
  * Hookah + COVID (Brian Primack)
    * Frequencies and subsamples currently running
    * Primack will want a descriptive sense of these data
  * Tobacco/Vape + Alcohol (Alex Russell)
    * Russell reviewed preliminary frequencies and subsamples
    * Final sample(s) will need to be run
  * Vape + Policy (Page Dobbs)
    * Dobbs has mostly-final subsamples for annotation
    * She may have additional data questions
