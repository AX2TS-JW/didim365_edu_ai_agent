const snakeToCamel = (str) => {
  return str.replace(/_([a-z])/g, (match, letter) => letter.toUpperCase());
};

const convertKeysToCamelCase = (obj) => {
  const newObj = {};
  Object.keys(obj).forEach(key => {
    newObj[snakeToCamel(key)] = obj[key];
  });
  return newObj;
};

export { snakeToCamel, convertKeysToCamelCase }