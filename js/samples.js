// Built-in sample data.
import {mulberry32,gaussFn,normalizeInPlace} from "./math.js";

// Synthetic "concept" vectors: each family and sub-family gets a random direction, and each word adds noise.
// A dependable first look at the tool; not the output of a trained model.
// ---------- Built-in demo: concept vectors (synthetic) ----------
const DEMO_SPEC={
  animals:{sub:{pet:"dog cat rabbit hamster parrot goldfish",farm:"cow sheep horse pig goat chicken",wild:"wolf lion tiger bear fox deer snake frog",bird:"eagle sparrow owl penguin",sea:"shark dolphin whale salmon octopus"}},
  food:{sub:{fruit:"apple banana orange mango grape strawberry lemon cherry",dish:"bread cheese rice pasta soup pizza salad noodles",sweet:"cake honey chocolate cookie",drink:"coffee tea juice milk"}},
  vehicles:{sub:{land:"car truck bus train bicycle motorcycle tram scooter",air:"airplane helicopter rocket drone",water:"boat ship submarine canoe"}},
  colors:{sub:{warm:"red yellow pink gold",cool:"blue green purple teal",neutral:"black white gray brown"}},
  emotions:{sub:{positive:"happy calm excited proud grateful hopeful joyful",negative:"sad angry afraid lonely anxious bored jealous"}},
  countries:{sub:{europe:"france germany norway spain italy",asia:"japan india vietnam china korea",americas:"brazil canada mexico peru chile",africa:"kenya egypt nigeria ghana"}},
  technology:{sub:{hardware:"computer laptop phone keyboard camera server chip",software:"software internet algorithm database browser app code"}},
  music:{sub:{instrument:"guitar piano violin drums trumpet flute cello",genre:"jazz rock opera blues hiphop reggae",idea:"melody rhythm song chord harmony"}},
};
const DEMO_CROSS={apple:{technology:0.45},orange:{colors:0.5},mouse:{technology:0.3},salmon:{food:0.55,colors:0.2},chicken:{food:0.5},gold:{technology:0.05},honey:{animals:0.15},drone:{technology:0.5},rocket:{technology:0.25},chip:{food:0.25},cookie:{technology:0.25},code:{music:0.05},blues:{colors:0.4,emotions:0.2},jazz:{emotions:0.1},cherry:{colors:0.35},lemon:{colors:0.25},green:{food:0.1},blue:{emotions:0.25},chile:{food:0.25},milk:{animals:0.2},harmony:{emotions:0.2}};
export function buildDemo(){
  const g=gaussFn(mulberry32(42));const D=48;const rv=(s)=>{const v=new Float64Array(D);for(let i=0;i<D;i++)v[i]=g()*s;return v;};
  const base={};for(const c in DEMO_SPEC)base[c]=rv(1);
  const rows=[];
  for(const c in DEMO_SPEC){for(const s in DEMO_SPEC[c].sub){const sv=rv(0.6);for(const w of DEMO_SPEC[c].sub[s].split(" ")){
    const v=rv(0.32);for(let i=0;i<D;i++)v[i]+=base[c][i]+sv[i];
    const cross=DEMO_CROSS[w]||{};for(const k in cross)for(let i=0;i<D;i++)v[i]+=base[k][i]*cross[k];
    rows.push({label:w,group:c,vector:Float32Array.from(normalizeInPlace(v))});
  }}}
  return rows;
}

export const SAMPLE_TOPICS=`[space] The telescope captured a faint galaxy billions of light years away.
[space] Astronauts on the space station grow lettuce in a small orbital garden.
[space] A new rocket engine was tested before the mission to Mars.
[space] Saturn's rings are made of countless pieces of ice and rock.
[space] The rover drilled into the Martian soil looking for signs of water.
[space] A solar eclipse darkened the sky over the coast for four minutes.
[space] Engineers are designing a lunar base that could house a small crew.
[space] The comet's tail stretched across the night sky for weeks.
[space] Satellites in low orbit help map storms and track wildfires.
[space] Scientists detected an exoplanet orbiting inside its star's habitable zone.
[cooking] Simmer the tomato sauce slowly with garlic and fresh basil.
[cooking] Knead the bread dough for ten minutes, then let it rise.
[cooking] Roast the vegetables in a hot oven until the edges caramelize.
[cooking] This recipe for lemon cake needs butter, sugar and three eggs.
[cooking] Season the soup with salt, pepper and a splash of cream.
[cooking] Sear the steak in a cast iron pan, then rest it before slicing.
[cooking] Toast the spices in a dry pan to bring out their aroma.
[cooking] Fold the beaten egg whites gently into the chocolate batter.
[cooking] Marinate the chicken overnight in yogurt, lemon and garlic.
[cooking] A pinch of salt makes caramel taste richer and less sweet.
[football] The striker scored twice in the second half to win the match.
[football] Fans sang loudly as the team walked onto the pitch.
[football] The goalkeeper saved a penalty in the final minute.
[football] The coach changed formation after the halftime whistle.
[football] A late red card left the defending side with ten players.
[football] The club signed a young midfielder for a record transfer fee.
[football] Rain made the pitch slippery and the passing sloppy.
[football] The league title will be decided on the final matchday.
[football] The referee checked the video replay before awarding the goal.
[football] Both teams pressed high from kickoff, and neither could score.
[money] The central bank raised interest rates to slow inflation.
[money] She moved her savings into a low cost index fund.
[money] Mortgage payments rose sharply after the rate increase.
[money] The company reported higher quarterly profits than analysts expected.
[money] Building an emergency fund can protect you from surprise bills.
[money] Stock markets fell as investors worried about a recession.
[money] Paying off credit card debt first saves the most interest.
[money] The startup raised new funding to expand into Europe.
[money] Rising prices squeezed household budgets this winter.
[money] A budget helps track where each paycheck actually goes.
[health] Regular walking lowers blood pressure and improves sleep.
[health] Drink water throughout the day to stay hydrated in the heat.
[health] The doctor recommended physical therapy for her knee injury.
[health] A balanced diet with vegetables supports a healthy heart.
[health] Getting enough sleep helps the immune system fight infections.
[health] Stretching after exercise can reduce muscle soreness.
[health] The clinic offers free flu vaccines every autumn.
[health] Chronic stress can raise blood pressure and disturb sleep.
[health] Physical activity improves mood as well as heart health.
[health] Wash your hands often to avoid catching a cold.
[music] The orchestra rehearsed the symphony late into the evening.
[music] He learned three guitar chords and wrote his first song.
[music] The jazz trio improvised over a slow, smoky rhythm.
[music] Her new album mixes folk melodies with electronic beats.
[music] The drummer kept a steady tempo through the long concert.
[music] Practicing scales every day made her piano playing smoother.
[music] The festival crowd sang every chorus with the band.
[music] A string quartet played softly at the wedding reception.
[music] The producer layered vocals and synths to build the track.
[music] Old vinyl records give the music a warm, crackling sound.`;

export const SAMPLE_REVIEWS=`[praise] The app is fast and the new design looks clean.
[praise] Setup took two minutes and everything just worked.
[praise] Support answered my question within an hour, very helpful.
[praise] I love how easy it is to share lists with my family.
[praise] Syncing between my phone and laptop is flawless now.
[praise] The dark mode is beautiful and easy on the eyes.
[praise] Best budgeting tool I have tried, worth every penny.
[praise] Great update, the search finally finds what I need.
[complaint] The app crashes every time I open the camera.
[complaint] I was charged twice this month and nobody replied.
[complaint] Login fails after the latest update, very frustrating.
[complaint] The screen freezes when I scroll through long lists.
[complaint] Notifications arrive hours late or not at all.
[complaint] Too many ads, it feels impossible to use without paying.
[complaint] My data disappeared after syncing, I lost a week of notes.
[complaint] Battery drains quickly while the app runs in the background.
[request] Please add an option to export my data as a spreadsheet.
[request] It would be great to have a widget for the home screen.
[request] Could you support recurring reminders on weekends only?
[request] I wish I could tag items with custom colors.
[request] Please add a calendar view to see tasks by week.
[request] An offline mode would make this perfect for travel.
[request] Let us sort lists alphabetically or by due date.
[request] Add two factor login for better account security.`;

