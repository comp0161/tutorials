// SuperCollider script for COMP0161 tutorial 8: Pitch

// ---------------------------------------------------------
// 0: setting up
// ---------------------------------------------------------

// use this to calibrate volume -- should be reasonably loud
{PinkNoise.ar(mul:1) !2}.play;

// for the time being base everything off middle C
// we'll look at tunings later, but for the moment
// just take advantage of SuperCollider's knowledge
(
~c4 = 60.midicps;

// some other basic config we'll use throughout
~ampl = 0.1;
~beat_freq = 1;
~lfo_rate = 1.0/20;
~lfo_range = 10;
~lfo = {LFTri.kr(~lfo_rate, 0, 1, 1) * ~lfo_range};
)


// ---------------------------------------------------------
// 1. beats, roughness, consonance and dissonance
// ---------------------------------------------------------

// for the time being we're going to just work with simple sine waves

// present the same single frequency on both channels
~single = {SinOsc.ar(~c4, 0, ~ampl, 0)!2}.play;
~single.release(2);

// sum two different frequencies and send to both channels
~double = {(SinOsc.ar(~c4, 0, ~ampl, 0) + SinOsc.ar(~c4 + ~beat_freq, 0, ~ampl, 0)) !2}.play;
~double.release(2);

// sum two frequencies with slow triangular modulation of one of them

~beater = {(SinOsc.ar(~c4, 0, ~ampl, 0) + SinOsc.ar(~c4 + ~lfo, 0, 0.1, 0)) !2}.play;
~beater.release(2);


// we can also do these things in stereo, with the frequencies on different sides
// - on speakers these should more or less merge at the listener to produce physical beats
// - on headphones "binaural beats" may be perceived even though the
//   physical sound waves are largely isolated from one another

~stereo = {[SinOsc.ar(~c4, 0, ~ampl, 0), SinOsc.ar(~c4 + ~beat_freq, 0, ~ampl, 0)]}.play;
~stereo.release(2);

~stereo_beater = {[SinOsc.ar(~c4, 0, ~ampl, 0), SinOsc.ar(~c4 + ~lfo, 0, ~ampl, 0)]}.play;
~stereo_beater.release(2);



// how about a more complex waveform?
// in this case we have to contend with beats from the partials too

~double = {(Saw.ar(~c4, ~ampl, 0) + Saw.ar(~c4 + ~beat_freq, ~ampl, 0)) !2}.play;
~double.release(2);

~beater = {(Saw.ar(~c4, ~ampl, 0) + Saw.ar(~c4 + ~lfo, ~ampl, 0)) !2}.play;
~beater.release(2);

~beater = {(Pulse.ar(~c4, 0.5, ~ampl, 0) + Pulse.ar(~c4 + ~lfo, 0.5, ~ampl, 0)) !2}.play;
~beater.release(2);


// the consonance or dissonance of two notes varies with their relative pitch
// seems to depend on their shared partials and the degrees of beating or roughness produced
// additionally, dissonance may be experienced when the sounds are distinct but not well
// separated by the auditory filters

// some examples presented in stereo

// ratios mostly taken from (some version of)
// 5-limit just intonation
(
~ratios = (root: 1, octave: 2, fifth: 3/2, fourth: 4/3,
	       aug_fourth: 45/32, tritone: 2.sqrt, dim_fifth: 64/45,
	       minor_third: 6/5, major_third: 5/4,
	       minor_second: 16/15, major_second: 9/8,
	       minor_sixth: 8/5, major_sixth: 5/3,
	       minor_seventh: 16/9, major_seventh: 15/8,
	       quarter_tone: 36/35, syntonic_comma: 81/80,
	       semi: 2.pow(1/12), whole: 2.pow(1/6), cent: 2.pow(1/1200));

~sine_interval = {
	arg left = \root, right, root = ~c4, ampl = ~ampl, ratios = ~ratios;
	var ratio_left, ratio_right;
	ratio_left = ratios[left];
	ratio_right = if(right.isNil, ratio_left, ratios[right]);
	{[ SinOsc.ar( root * ratio_left, 0, ampl, 0 ),
	   SinOsc.ar( root * ratio_right, 0, ampl, 0 ) ]}.play;
};

~saw_interval = {
	arg left = \root, right, root = ~c4, ampl = ~ampl, ratios = ~ratios;
	var ratio_left, ratio_right;
	ratio_left = ratios[left];
	ratio_right = if(right.isNil, ratio_left, ratios[right]);
	{[ Saw.ar( root * ratio_left, ampl, 0 ),
	   Saw.ar( root * ratio_right, ampl, 0 ) ]}.play;
};

~square_interval = {
	arg left = \root, right, root = ~c4, ampl = ~ampl, ratios = ~ratios;
	var ratio_left, ratio_right;
	ratio_left = ratios[left];
	ratio_right = if(right.isNil, ratio_left, ratios[right]);
	{[ Pulse.ar( root * ratio_left, 0.5, ampl, 0 ),
	   Pulse.ar( root * ratio_right, 0.5, ampl, 0 ) ]}.play;
};
)

~sine_interval.value(\root);
~sine_interval.value(\octave);
~sine_interval.value(\root, \octave);

~sine_interval.value(\root);
~sine_interval.value(\fifth);
~sine_interval.value(\root, \fifth);

~sine_interval.value(\root);
~sine_interval.value(\fourth);
~sine_interval.value(\root, \fourth);

~sine_interval.value(\root);
~sine_interval.value(\major_third);
~sine_interval.value(\root, \major_third);

~sine_interval.value(\root);
~sine_interval.value(\minor_third);
~sine_interval.value(\root, \minor_third);

~sine_interval.value(\root);
~sine_interval.value(\major_second);
~sine_interval.value(\root, \major_second);

~sine_interval.value(\root);
~sine_interval.value(\minor_second);
~sine_interval.value(\root, \minor_second);

~sine_interval.value(\root);
~sine_interval.value(\minor_sixth);
~sine_interval.value(\root, \minor_sixth);

~sine_interval.value(\root);
~sine_interval.value(\major_sixth);
~sine_interval.value(\root, \major_sixth);

~sine_interval.value(\root);
~sine_interval.value(\minor_seventh);
~sine_interval.value(\root, \minor_seventh);

~sine_interval.value(\root);
~sine_interval.value(\major_seventh);
~sine_interval.value(\root, \major_seventh);

~sine_interval.value(\root);
~sine_interval.value(\aug_fourth);
~sine_interval.value(\root, \aug_fourth);

~sine_interval.value(\root);
~sine_interval.value(\dim_fifth);
~sine_interval.value(\root, \dim_fifth);

~sine_interval.value(\aug_fourth);
~sine_interval.value(\dim_fifth);
~sine_interval.value(\aug_fourth, \dim_fifth);

~sine_interval.value(\root);
~sine_interval.value(\tritone);
~sine_interval.value(\root, \tritone);

~sine_interval.value(\root);
~sine_interval.value(\quarter_tone);
~sine_interval.value(\root, \quarter_tone);

~sine_interval.value(\root);
~sine_interval.value(\syntonic_comma);
~sine_interval.value(\root, \syntonic_comma);

// again, the effect is slightly different with more complex waveforms
// because of the beating of the partials

~saw_interval.value(\root);
~saw_interval.value(\octave);
~saw_interval.value(\root, \octave);

~saw_interval.value(\root);
~saw_interval.value(\fifth);
~saw_interval.value(\root, \fifth);

~saw_interval.value(\root);
~saw_interval.value(\fourth);
~saw_interval.value(\root, \fourth);

~saw_interval.value(\root);
~saw_interval.value(\major_third);
~saw_interval.value(\root, \major_third);

~saw_interval.value(\root);
~saw_interval.value(\minor_third);
~saw_interval.value(\root, \minor_third);

~saw_interval.value(\root);
~saw_interval.value(\major_second);
~saw_interval.value(\root, \major_second);

~saw_interval.value(\root);
~saw_interval.value(\minor_second);
~saw_interval.value(\root, \minor_second);

~saw_interval.value(\root);
~saw_interval.value(\minor_sixth);
~saw_interval.value(\root, \minor_sixth);

~saw_interval.value(\root);
~saw_interval.value(\major_sixth);
~saw_interval.value(\root, \major_sixth);

~saw_interval.value(\root);
~saw_interval.value(\minor_seventh);
~saw_interval.value(\root, \minor_seventh);

~saw_interval.value(\root);
~saw_interval.value(\major_seventh);
~saw_interval.value(\root, \major_seventh);

~saw_interval.value(\root);
~saw_interval.value(\aug_fourth);
~saw_interval.value(\root, \aug_fourth);

~saw_interval.value(\root);
~saw_interval.value(\dim_fifth);
~saw_interval.value(\root, \dim_fifth);

~saw_interval.value(\aug_fourth, \dim_fifth);

~saw_interval.value(\root);
~saw_interval.value(\tritone);
~saw_interval.value(\root, \tritone);

~saw_interval.value(\root);
~saw_interval.value(\quarter_tone);
~saw_interval.value(\root, \quarter_tone);

~saw_interval.value(\root);
~saw_interval.value(\syntonic_comma);
~saw_interval.value(\root, \syntonic_comma);



~square_interval.value(\root);
~square_interval.value(\octave);
~square_interval.value(\root, \octave);

~square_interval.value(\root);
~square_interval.value(\fifth);
~square_interval.value(\root, \fifth);

~square_interval.value(\root);
~square_interval.value(\fourth);
~square_interval.value(\root, \fourth);

~square_interval.value(\root);
~square_interval.value(\major_third);
~square_interval.value(\root, \major_third);

~square_interval.value(\root);
~square_interval.value(\minor_third);
~square_interval.value(\root, \minor_third);

~square_interval.value(\root);
~square_interval.value(\major_second);
~square_interval.value(\root, \major_second);

~square_interval.value(\root);
~square_interval.value(\minor_second);
~square_interval.value(\root, \minor_second);

~square_interval.value(\root);
~square_interval.value(\minor_sixth);
~square_interval.value(\root, \minor_sixth);

~square_interval.value(\root);
~square_interval.value(\major_sixth);
~square_interval.value(\root, \major_sixth);

~square_interval.value(\root);
~square_interval.value(\minor_seventh);
~square_interval.value(\root, \minor_seventh);

~square_interval.value(\root);
~square_interval.value(\major_seventh);
~square_interval.value(\root, \major_seventh);

~square_interval.value(\root);
~square_interval.value(\aug_fourth);
~square_interval.value(\root, \aug_fourth);

~square_interval.value(\root);
~square_interval.value(\dim_fifth);
~square_interval.value(\root, \dim_fifth);

~square_interval.value(\aug_fourth, \dim_fifth);

~square_interval.value(\root);
~square_interval.value(\tritone);
~square_interval.value(\root, \tritone);

~square_interval.value(\root);
~square_interval.value(\quarter_tone);
~square_interval.value(\root, \quarter_tone);

~square_interval.value(\root);
~square_interval.value(\syntonic_comma);
~square_interval.value(\root, \syntonic_comma);


// 2. tuning & temperament

Tuning.directory;
Scale.directory;

// middle is midi note 60
60.midicps;

// A is sixth of C major
Scale.major.degreeToFreq(5, 60.midicps, 0).round;

// middle C is third of A minor, one octave down from 440
Scale.minor.degreeToFreq(2, 440.0, -1);

// just ratios vs 12tet ratios
Scale.major(\just).ratios;
Scale.major(\et12).ratios;

Scale.chromatic(\just).ratios;
Scale.chromatic(\et12).ratios;

// some scales from middle C
(
Pbind(\freq, Pseq(Scale.major(\just).degreeToFreq([0,1,2,3,4,5,6,7], 60.midicps, 0)), \dur, 0.35).play;
Pbind(\freq, Pseq(Scale.major(\et12).degreeToFreq([0,1,2,3,4,5,6,7], 60.midicps, 0)), \dur, 0.35).play;
)

(
Pbind(\freq, Pseq(Scale.minor(\just).degreeToFreq([0,1,2,3,4,5,6,7], 60.midicps, 0)), \dur, 0.35).play;
Pbind(\freq, Pseq(Scale.minor(\et12).degreeToFreq([0,1,2,3,4,5,6,7], 60.midicps, 0)), \dur, 0.35).play;
)

(
Pbind(\freq, Pseq(Scale.chromatic(\just).degreeToFreq([0,1,2,3,4,5,6,7,8,9,10,11,12], 60.midicps, 0)), \dur, 0.35).play;
Pbind(\freq, Pseq(Scale.chromatic(\et12).degreeToFreq([0,1,2,3,4,5,6,7,8,9,10,11,12], 60.midicps, 0)), \dur, 0.35).play;
)

// arpeggios
(
Pbind(\freq, Pseq(Scale.major(\just).degreeToFreq([0,2,4,7,4,2,0], 60.midicps, 0)), \dur, 0.45).play;
Pbind(\freq, Pseq(Scale.major(\et12).degreeToFreq([0,2,4,7,4,2,0], 60.midicps, 0)), \dur, 0.45).play;
)

// triads

// C major triad in just intonation tuned to C and 12TET
(
Pbind(\freq, Scale.chromatic(\just).degreeToFreq([0,4,7], 60.midicps, 0), \dur, 0.5).play;
Pbind(\freq, Scale.chromatic(\et12).degreeToFreq([0,4,7], 60.midicps, 0), \dur, 0.5).play;
)

// C# major chord in just intonation tuned to C and C# and also 12TET
(
Pbind(\freq, Scale.chromatic(\just).degreeToFreq([1,5,8], 60.midicps, 0), \dur, 0.5).play;
Pbind(\freq, Scale.chromatic(\just).degreeToFreq([0,4,7], 61.midicps, 0), \dur, 0.5).play;
Pbind(\freq, Scale.chromatic(\et12).degreeToFreq([1,5,8], 60.midicps, 0), \dur, 0.5).play;
)

// F# major chord in just intonation tuned to C and F# and also 12TET
(
Pbind(\freq, Scale.chromatic(\just).degreeToFreq([6,10,13], 60.midicps, 0), \dur, 0.5).play;
Pbind(\freq, Scale.chromatic(\just).degreeToFreq([0,4,7], 66.midicps, 0), \dur, 0.5).play;
Pbind(\freq, Scale.chromatic(\et12).degreeToFreq([6,10,13], 60.midicps, 0), \dur, 0.5).play;
)

// Bb major chord in just intonation tuned to C and Bb and also 12TET
(
Pbind(\freq, Scale.chromatic(\just).degreeToFreq([10,14,17], 60.midicps, 0), \dur, 0.5).play;
Pbind(\freq, Scale.chromatic(\just).degreeToFreq([0,4,7], 70.midicps, 0), \dur, 0.5).play;
Pbind(\freq, Scale.chromatic(\et12).degreeToFreq([10,14,17], 60.midicps, 0), \dur, 0.5).play;
)
